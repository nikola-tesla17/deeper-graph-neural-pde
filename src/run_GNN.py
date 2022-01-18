import os
import argparse
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from GNN import GNN
from GNN_early import GNNEarly
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
import time, datetime
from data import get_dataset, set_train_val_test_split
from ogb.nodeproppred import Evaluator
from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling
from best_params import best_params_dict
from greed_params import greed_test_params, greed_run_params, greed_hyper_params, greed_ablation_params, tf_ablation_args, not_sweep_args
import wandb

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, optimizer, data, pos_encoding=None):

  lf = torch.nn.functional.nll_loss if model.opt['dataset'] == 'ogbn-arxiv' else torch.nn.CrossEntropyLoss()

  if model.opt['wandb_watch_grad']: # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, lf, log="all", log_freq=10)

  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    # lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    # lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  model.fm.update(model.getNFE())
  model.resetNFE()
  # torch.autograd.set_detect_anomaly(True)
  loss.backward()#retain_graph=True)
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


def train_OGB(model, mp, optimizer, data, pos_encoding=None):

  lf = torch.nn.functional.nll_loss if model.opt['dataset'] == 'ogbn-arxiv' else torch.nn.CrossEntropyLoss()

  if model.opt['wandb_watch_grad']: # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, lf, log="all", log_freq=10)

  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  pos_encoding = mp(pos_encoding).to(model.device)
  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    # lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    # lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat, pos_encoding), []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


@torch.no_grad()
def test_OGB(model, data, pos_encoding, opt):
  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'

  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

  evaluator = Evaluator(name=name)
  model.eval()

  out = model(feat, pos_encoding).log_softmax(dim=-1)
  y_pred = out.argmax(dim=-1, keepdim=True)

  train_acc = evaluator.eval({
    'y_true': data.y[data.train_mask],
    'y_pred': y_pred[data.train_mask],
  })['acc']
  valid_acc = evaluator.eval({
    'y_true': data.y[data.val_mask],
    'y_pred': y_pred[data.val_mask],
  })['acc']
  test_acc = evaluator.eval({
    'y_true': data.y[data.test_mask],
    'y_pred': y_pred[data.test_mask],
  })['acc']

  return train_acc, valid_acc, test_acc


def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['beltrami']:
    opt['beltrami'] = True
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] != 1:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']


def main(cmd_opt):

  if cmd_opt['use_best_params']:
    best_opt = best_params_dict[cmd_opt['dataset']]
    opt = {**cmd_opt, **best_opt}
    merge_cmd_args(cmd_opt, opt)
  else:
    opt = cmd_opt

  if opt['wandb']:
    if opt['use_wandb_offline']:
      os.environ["WANDB_MODE"] = "offline"
    else:
      os.environ["WANDB_MODE"] = "run"
  else:
    os.environ["WANDB_MODE"] = "disabled"  # sets as NOOP, saves keep writing: if opt['wandb']:

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt['device'] = device


  if 'wandb_run_name' in opt.keys():
    wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
               name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)
  else:
    wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
               reinit=True, config=opt, allow_val_change=True) #required when update hidden_dim in beltrami

  # wandb.config.update(opt, allow_val_change=True) #required when update hidden_dim in beltrami
  opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!


  wandb.define_metric("epoch_step") #Customize axes - https://docs.wandb.ai/guides/track/log
  if opt['wandb_track_grad_flow']:
    wandb.define_metric("grad_flow_step") #Customize axes - https://docs.wandb.ai/guides/track/log
    wandb.define_metric("gf_e*", step_metric="grad_flow_step") #grad_flow_epoch*


  dataset = get_dataset(opt, '../data', opt['not_lcc'])
  if opt['beltrami']:
    pos_encoding = apply_beltrami(dataset.data, opt).to(device)
    opt['pos_enc_dim'] = pos_encoding.shape[1]
  else:
    pos_encoding = None

  if opt['rewire_KNN'] or opt['fa_layer']:
    model = GNN_KNN(opt, dataset, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, dataset, device).to(
      device)
  else:
    model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)

  if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
    dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                            num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  data = dataset.data.to(device)

  parameters = [p for p in model.parameters() if p.requires_grad]
  print(opt)
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_time = best_epoch = train_acc = val_acc = test_acc = 0

  this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test

  for epoch in range(1, opt['epoch']):
    start_time = time.time()

    if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
      ei = apply_KNN(data, pos_encoding, model, opt)
      model.odeblock.odefunc.edge_index = ei

    loss = train(model, optimizer, data, pos_encoding)
    model.odeblock.odefunc.wandb_step = 0 # resets the wandstep in function after train forward pass

    tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)
    model.odeblock.odefunc.wandb_step = 0 # resets the wandstep in function after eval forward pass

    best_time = opt['time']
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
      best_time = opt['time']
    if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
      best_epoch = epoch
      val_acc = model.odeblock.test_integrator.solver.best_val
      test_acc = model.odeblock.test_integrator.solver.best_test
      train_acc = model.odeblock.test_integrator.solver.best_train
      best_time = model.odeblock.test_integrator.solver.best_time

    if ((epoch) % opt['wandb_log_freq']) == 0:
      wandb.log({"loss": loss,
                 # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
                 "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "epoch_step": epoch}) #, step=epoch) wandb: WARNING Step must only increase in log calls

    print(f"Epoch: {epoch}, Runtime: {time.time() - start_time:.3f}, Loss: {loss:.3f}, "
          f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
          f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")
    if opt['function'] == 'greed':
      model.odeblock.odefunc.epoch += 1

  print(f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")
  # https://docs.wandb.ai/guides/track/log
  # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
  # https://docs.wandb.ai/ref/app/features/custom-charts/walkthrough
  #todo Customize axes - https://docs.wandb.ai/guides/track/log

  # wandb.log({'final_test_accuracy': test_acc, 'final_val_accuracy': val_acc, 'final_loss': loss, 'best_epoch': best_epoch}) #For values that are logged with wandb.log, we automatically set summary to the last value added
  wandb_run.finish()
  return train_acc, val_acc, test_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  # data args
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
  parser.add_argument('--planetoid_split', action='store_true',
                      help='use planetoid splits for Cora/Citeseer/Pubmed')
  # GNN args
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                      help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, help='constant, mixed, attention, hard_attention')
  parser.add_argument('--function', type=str, help='laplacian, transformer, greed, GAT')
  # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
  #                     help='Add a fully connected layer to the encoder.')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')

  # ODE args
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=0.1,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
  parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                      help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
  parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                      help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument("--max_nfe", type=int, default=1000,
                      help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  # parser.add_argument("--no_early", action="store_true",
  #                     help="Whether or not to use early stopping of the ODE integrator when testing.")
  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
  parser.add_argument("--max_test_steps", type=int, default=100,
                      help="Maximum number steps for the dopri5Early test integrator. "
                           "used if getting OOM errors at test time")

  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,pearson, exp_kernel")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # rewiring args
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                      help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  parser.add_argument('--att_samp_pct', type=float, default=1,
                      help="float in [0,1). The percentage of edges to retain based on attention scores")
  parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                      help='incorporate the feature grad in attention based edge dropout')
  parser.add_argument("--exact", action="store_true",
                      help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
  parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
  parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
  parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
  parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
  parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
  parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
  parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
  parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
  parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
  # beltrami args
  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
  parser.add_argument('--pos_enc_type', type=str, default="DW64",
                      help='positional encoder either GDC, DW64, DW128, DW256')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
  parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
  parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                      help="random, ,anchored, importance, degree")
  parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
  parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--edge_sampling_space', type=str, default="attention",
                      help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
  parser.add_argument('--symmetric_attention', action='store_true',
                      help='maks the attention symmetric for rewring in QK space')
  parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
  parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")
  parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

  # wandb logging and tuning
  parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
  parser.add_argument('-wandb_offline', dest='use_wandb_offline', action='store_true')  # https://docs.wandb.ai/guides/technical-faq

  parser.add_argument('--wandb_sweep', action='store_true', help="flag if sweeping") #if not it picks up params in greed_params
  parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
  parser.add_argument('--wandb_track_grad_flow', action='store_true')

  parser.add_argument('--wandb_entity', default="graph_neural_diffusion", type=str,
                      help="jrowbottomwnb, ger__man")  # not used as default set in web browser settings
  parser.add_argument('--wandb_project', default="greed", type=str)
  parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
  parser.add_argument('--wandb_run_name', default=None, type=str)
  parser.add_argument('--wandb_output_dir', default='./wandb_output',
                      help='folder to output results, images and model checkpoints')
  parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
  parser.add_argument('--wandb_epoch_list', nargs='+',  default=[0, 1, 2, 4, 8, 16], help='list of epochs to log gradient flow')

  #wandb setup sweep args
  parser.add_argument('--tau_reg', type=int, default=2)
  parser.add_argument('--test_mu_0', type=str, default='True') #action='store_true')
  parser.add_argument('--test_no_chanel_mix', type=str, default='True') #action='store_true')
  parser.add_argument('--test_omit_metric', type=str, default='True') #action='store_true')
  parser.add_argument('--test_tau_remove_tanh', type=str, default='True') #action='store_true')
  parser.add_argument('--test_tau_symmetric', type=str, default='True') #action='store_true')
  parser.add_argument('--test_tau_outside', type=str, default='True') #action='store_true')
  parser.add_argument('--test_linear_L0', type=str, default='True') #action='store_true')
  parser.add_argument('--test_R1R2_0', type=str, default='True') #action='store_true')

  # Temp changing these to be strings so can tune over
  # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
  #                     help='Add a fully connected layer to the encoder.')
  # parser.add_argument("--no_early", action="store_true",
  #                     help="Whether or not to use early stopping of the ODE integrator when testing.")

  parser.add_argument('--use_mlp', type=str, default='False') #action='store_true')
  parser.add_argument('--no_early', type=str, default='False') #action='store_true')

  #greed args
  parser.add_argument('--use_best_params', action='store_true', help="flag to take the best BLEND params")
  parser.add_argument('--greed_momentum', action='store_true', help="flag to use momentum grad flow")
  parser.add_argument('--momentum_alpha', type=float, default=0.2, help="alpha to use in momentum grad flow")
  parser.add_argument('--dim_p_omega', type=int, default=16, help="inner dimension for Omega")
  parser.add_argument('--dim_p_w', type=int, default=16, help="inner dimension for W")
  parser.add_argument('--gamma_epsilon', type=float, default=0.01, help="epsilon value used for numerical stability in get_gamma")

  args = parser.parse_args()
  opt = vars(args)

  if opt['function'] == 'greed' or opt['function'] == 'greed_scaledDP' or opt['function'] == 'greed_linear':
    opt = greed_run_params(opt)  ###basic params for GREED

    if not opt['wandb_sweep']: #sweeps are run from YAML config so don't need these
      opt = not_sweep_args(opt, project_name='greed_runs', group_name='testing')
      # # args for running locally - specified in YAML for tunes
      # opt['wandb'] = True
      # opt['wandb_track_grad_flow'] = False # don't plot grad flows when tuning
      # opt['wandb_project'] = "greed_runs"
      # opt['wandb_group'] = "testing" #"tuning" eval
      # DT = datetime.datetime.now()
      # opt['wandb_run_name'] = DT.strftime("%m%d_%H%M%S_") + "wandb_best_BLEND_params"#"wandb_log_gradflow_test3"
      # #hyper-params
      # if not opt['use_best_params']:
      #   opt = greed_hyper_params(opt)
      # opt = greed_ablation_params(opt)

    opt = tf_ablation_args(opt)

  main(opt)

#terminal commands for sweeps
#wandb sweep ../wandb_sweep_configs/greed_sweep_grid.yaml
#./run_sweeps.sh XXX