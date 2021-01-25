import argparse
import torch
import os
from GNN_image import GNN_image
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch_geometric.utils import to_dense_adj
import pandas as pd
from data_image import load_data
from image_opt import get_image_opt


def UnNormalizeCIFAR(data):
  #normalises each image channel to range [0,1] from [-1, 1]
  # return (data - torch.amin(data,dim=(0,1))) / (torch.amax(data,dim=(0,1)) - torch.amin(data,dim=(0,1)))
  return data * 0.5 + 0.5

@torch.no_grad()
def plot_image_T(model, dataset, opt, modelpath, height=2, width=3):
  loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
  fig = plt.figure() #figsize=(width*10, height*10))
  for batch_idx, batch in enumerate(loader):
    out = model.forward_plot_T(batch.x)
    break
  for i in range(height*width):
    # t == 0
    plt.subplot(2*height, width, i + 1)
    plt.tight_layout()
    plt.axis('off')
    mask = batch.batch == i
    A = batch.x[torch.nonzero(mask)].squeeze()
    A = A.view(opt['im_height'], opt['im_width'], opt['im_chan'])
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title("t=0 Ground Truth: {}".format(batch.y[i].item()))

    #t == T
    plt.subplot(2*height, width, height*width + i + 1)
    plt.tight_layout()
    plt.axis('off')
    A = out[torch.nonzero(mask)].squeeze()
    A = A.view(model.opt['im_height'], model.opt['im_width'], model.opt['im_chan'])
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title("t=T Ground Truth: {}".format(batch.y[i].item()))
  return fig


@torch.no_grad()
def create_animation_old(model, dataset, opt, height, width, frames):
  loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
  for batch_idx, batch in enumerate(loader):
    paths = model.forward_plot_path(batch.x, frames)
    break
  # draw graph initial graph
  fig = plt.figure()
  for i in range(height * width):
    plt.subplot(height, width, i + 1)
    plt.tight_layout()
    # mask = batch.batch == i
    A = paths[i, 0, :].view(opt['im_height'], opt['im_width'], opt['im_chan'])
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation='none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A)
    plt.title("t=0 Ground Truth: {}".format(batch.y[i].item()))
    plt.axis('off')
  # loop through data and update plot
  def update(ii):
    for i in range(height * width):
      plt.subplot(height, width, i + 1)
      plt.tight_layout()
      A = paths[i, ii, :].view(model.opt['im_height'], model.opt['im_width'], model.opt['im_chan'])
      if opt['im_dataset'] == 'MNIST':
        plt.imshow(A, cmap='gray', interpolation='none')
      elif opt['im_dataset'] == 'CIFAR':
        A = UnNormalizeCIFAR(A)
        plt.imshow(A)
      plt.title("t={} Ground Truth: {}".format(ii, batch.y[i].item()))
      plt.axis('off')
  fig = plt.gcf()
  animation = FuncAnimation(fig, func=update, frames=frames)#, blit=True)
  return animation


@torch.no_grad()
def create_pixel_intensity_old(model, dataset, opt, height, width, frames):
  # max / min intensity plot
  loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
  for batch_idx, batch in enumerate(loader):
    paths = model.forward_plot_path(batch.x, frames)
    break
  # draw graph initial graph
  fig = plt.figure() #figsize=(width*10, height*10))
  for i in range(height * width):
    plt.subplot(height, width, i + 1)
    plt.tight_layout()
    if opt['im_dataset'] == 'MNIST':
      A = paths[i, :, :]
      plt.plot(torch.max(A,dim=1)[0], color='red')
      plt.plot(torch.min(A,dim=1)[0], color='green')
      plt.plot(torch.mean(A,dim=1), color='blue')
    elif opt['im_dataset'] == 'CIFAR':
      A = paths[i,:,:].view(paths.shape[1], opt['im_height'] * opt['im_width'], opt['im_chan'])
      plt.plot(torch.max(A, dim=1)[0][:,0],color='red')
      plt.plot(torch.max(A, dim=1)[0][:,1],color='green')
      plt.plot(torch.max(A, dim=1)[0][:,2],color='blue')
      plt.plot(torch.min(A, dim=1)[0][:,0],color='red')
      plt.plot(torch.min(A, dim=1)[0][:,1],color='green')
      plt.plot(torch.min(A, dim=1)[0][:,2],color='blue')
      plt.plot(torch.mean(A, dim=1)[:,0],color='red')
      plt.plot(torch.mean(A, dim=1)[:,1],color='green')
      plt.plot(torch.mean(A, dim=1)[:,2],color='blue')
    plt.title("Evolution of Pixel Intensity, Ground Truth: {}".format(batch.y[i].item()))
  return fig


@torch.no_grad()
def plot_att_heat(model, model_key, modelpath):
  #visualisation of ATT for the 1st image in the batch
  im_height = model.opt['im_height']
  im_width = model.opt['im_width']
  im_chan = model.opt['im_chan']
  hwc = im_height * im_width
  edge_index = model.odeblock.odefunc.edge_index
  num_nodes = model.opt['num_nodes']
  batch_size = model.opt['batch_size']
  edge_weight = model.odeblock.odefunc.edge_weight
  dense_att = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight,
                           max_num_nodes=num_nodes*batch_size)[0,:num_nodes,:num_nodes]
  square_att = dense_att.view(num_nodes, num_nodes)
  x_np = square_att.numpy()
  x_df = pd.DataFrame(x_np)
  x_df.to_csv(f"{modelpath}_att.csv")
  fig = plt.figure()
  plt.tight_layout()
  plt.imshow(square_att, cmap='hot', interpolation='nearest')
  plt.title("Attention Heat Map {}".format(model_key))
  return fig
  # useful code to overcome normalisation colour bar
  # https: // matplotlib.org / 3.3.3 / gallery / images_contours_and_fields / multi_image.html  # sphx-glr-gallery-images-contours-and-fields-multi-image-py


@torch.no_grad()
def plot_image(labels, paths, time, opt, pic_folder, samples):
  savefolder = f"{pic_folder}/image_{time}"
  try:
    os.mkdir(savefolder)
  except OSError:
    print("Creation of the directory %s failed" % savefolder)
  else:
    print("Successfully created the directory %s " % savefolder)
  for i in range(samples):
    fig = plt.figure()
    plt.tight_layout()
    plt.axis('off')
    A = paths[i,time,:].view(opt['im_height'], opt['im_width'], opt['im_chan'])
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title(f"t={time} Ground Truth: {labels[i].item()}")
    plt.savefig(f"{savefolder}/image_{time}_{i}.png", format="PNG")
  return fig


@torch.no_grad()
def create_animation(labels, paths, frames, fps, opt, pic_folder, samples):
  savefolder = f"{pic_folder}/animations"
  try:
    os.mkdir(savefolder)
  except OSError:
    print("Creation of the directory %s failed" % savefolder)
  else:
    print("Successfully created the directory %s " % savefolder)
  # draw graph initial graph
  for i in range(samples):
    fig = plt.figure()
    plt.tight_layout()
    plt.axis('off')
    A = paths[i,0,:].view(opt['im_height'], opt['im_width'], opt['im_chan'])

    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title("t=0 Ground Truth: {}".format(labels[i].item()))
    # loop through data and update plot
    def update(ii):
      plt.tight_layout()
      A = paths[i,ii,:].view(opt['im_height'], opt['im_width'], opt['im_chan'])
      if opt['im_dataset'] == 'MNIST':
        plt.imshow(A, cmap='gray', interpolation = 'none')
      elif opt['im_dataset'] == 'CIFAR':
        A = UnNormalizeCIFAR(A)
        plt.imshow(A, interpolation = 'none')
      plt.title(f"t={ii} Ground Truth: {labels[i].item()}")
    fig = plt.gcf()
    animation = FuncAnimation(fig, func=update, frames=frames)
    animation.save(f'{savefolder}/animation{i}.gif', fps=fps)#, writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=fps)
  # return animation


@torch.no_grad()
def create_pixel_intensity(labels, paths, opt, pic_folder, samples):
  savefolder = f"{pic_folder}/maxmin"
  try:
    os.mkdir(savefolder)
  except OSError:
    print("Creation of the directory %s failed" % savefolder)
  else:
    print("Successfully created the directory %s " % savefolder)
  for i in range(samples):
    fig = plt.figure()
    plt.tight_layout()
    plt.axis('off')
    if opt['im_dataset'] == 'MNIST':
      A = paths[i, :, :]
      plt.plot(torch.max(A, dim=1)[0], color='red')
      plt.plot(torch.min(A, dim=1)[0], color='green')
      plt.plot(torch.mean(A, dim=1), color='blue')
    elif opt['im_dataset'] == 'CIFAR':
      A = paths[i, :, :].view(paths.shape[1], opt['im_height'] * opt['im_width'], opt['im_chan'])
      plt.plot(torch.max(A, dim=1)[0][:, 0], color='red')
      plt.plot(torch.max(A, dim=1)[0][:, 1], color='green')
      plt.plot(torch.max(A, dim=1)[0][:, 2], color='blue')
      plt.plot(torch.min(A, dim=1)[0][:, 0], color='red')
      plt.plot(torch.min(A, dim=1)[0][:, 1], color='green')
      plt.plot(torch.min(A, dim=1)[0][:, 2], color='blue')
      plt.plot(torch.mean(A, dim=1)[:, 0], color='red')
      plt.plot(torch.mean(A, dim=1)[:, 1], color='green')
      plt.plot(torch.mean(A, dim=1)[:, 2], color='blue')
    plt.title("Max/Min, Ground Truth: {}".format(labels[i].item()))
    plt.savefig(f"{savefolder}/max_min_{i}.png", format="PNG")

  return fig


def build_all(model_keys, frames = 10, samples=6):
  directory = f"../models/"
  df = pd.read_csv(f'{directory}models.csv')
  for model_key in model_keys:
    for filename in os.listdir(directory):
      if filename.startswith(model_key):
        path = os.path.join(directory, filename)
        print(path)
        break
    [_, _, data_name, blck, fct] = path.split("_")
    modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
    modelpath = f"{modelfolder}/model_{model_key}"
    optdf = df[df.model_key == model_key]
    intcols = ['num_class','im_chan','im_height','im_width','num_nodes']
    optdf[intcols].astype(int)
    opt = optdf.to_dict('records')[0]
    ###load data and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_train, data_test = load_data(opt)
    loader = DataLoader(data_train, batch_size=opt['batch_size'], shuffle=True)
    for batch_idx, batch in enumerate(loader):
        break
    batch.to(device)
    edge_index_gpu = batch.edge_index
    edge_attr_gpu = batch.edge_attr
    if edge_index_gpu is not None: edge_index_gpu.to(device)
    if edge_attr_gpu is not None: edge_index_gpu.to(device)

    opt['time'] = opt['time'] / frames
    model = GNN_image(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                      batch.edge_attr, device).to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.to(device)
    model.eval()
    ###do forward pass
    for batch_idx, batch in enumerate(loader):
      batch_paths = model.forward_plot_path(batch.x, 2*frames)
      break
    plot_image(batch.y, batch_paths, time=0, opt=opt, pic_folder=modelfolder, samples=samples)
    plot_image(batch.y, batch_paths, time=10, opt=opt, pic_folder=modelfolder, samples=samples)
    create_animation(batch.y, batch_paths, 2*frames, fps=0.75, opt=opt, pic_folder=modelfolder, samples=samples)
    create_pixel_intensity(batch.y, batch_paths, opt, pic_folder=modelfolder, samples=samples)

def main(model_keys):
  for model_key in model_keys:
  # model_key = '20210124_202732' #'20210121_200920'#
    directory = f"../models/"
    for filename in os.listdir(directory):
      if filename.startswith(model_key):
        path = os.path.join(directory, filename)
        print(path)
        break
    [_, _, data_name, blck, fct] = path.split("_")

    modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
    modelpath = f"{modelfolder}/model_{model_key}"

    df = pd.read_csv(f'{directory}models.csv')
    optdf = df[df.model_key == model_key]
    intcols = ['num_class','im_chan','im_height','im_width','num_nodes']
    optdf[intcols].astype(int)
    opt = optdf.to_dict('records')[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Data")
    data_train, data_test = load_data(opt)
    print("creating GNN model")
    loader = DataLoader(data_train, batch_size=opt['batch_size'], shuffle=True)
    for batch_idx, batch in enumerate(loader):
        break
    batch.to(device)
    edge_index_gpu = batch.edge_index
    edge_attr_gpu = batch.edge_attr
    if edge_index_gpu is not None: edge_index_gpu.to(device)
    if edge_attr_gpu is not None: edge_index_gpu.to(device)

    N = 10
    opt['time'] = opt['time'] / N
    model = GNN_image(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                      batch.edge_attr, device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.to(device)
    model.eval()

    # # 1)
    fig = plot_image_T(model, data_test, opt, modelpath, height=2, width=3)
    plt.savefig(f"{modelpath}_imageT.png", format="PNG")
    # 2)
    animation = create_animation_old(model, data_test, opt, height=2, width=3, frames=10)
    # animation.save(f'{modelpath}_animation.gif', writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=2)
    # animation.save(f'{modelpath}_animation2.gif', fps=2)

    # from IPython.display import HTML
    # HTML(animation.to_html5_video())
    plt.rcParams['animation.ffmpeg_path'] = '/home/jr1419home/anaconda3/envs/GNN_WSL/bin/ffmpeg'
    animation.save(f'{modelpath}_animation3.mp4', writer='ffmpeg', fps=2)
    # 3)
    # fig = plot_att_heat(model, model_key, modelpath)
    # plt.savefig(f"{modelpath}_AttHeat.png", format="PNG")
    # # 4)
    fig = create_pixel_intensity_old(model, data_test, opt, height=2, width=3, frames=10)
    plt.savefig(f"{modelpath}_pixel_intensity.png", format="PNG")

if __name__ == '__main__':
  # model_keys = ['20210125_002517', '20210125_002603']
  # model_keys = ['20210125_111920', '20210125_115601']
  model_keys = ['20210125_115601']
  main(model_keys)
  # build_all(model_keys)