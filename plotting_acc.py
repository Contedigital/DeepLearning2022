import os
import numpy as np
from torch import Tensor
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
from torchvision.utils import save_image


def plot_autoencoder_stats(x, x_hat, x_b, z, epoch, train_loss, valid_loss, valid_acc) -> None:
    """
    An utility 
    """
    # -- Plotting --
    f = plt.figure(figsize=(14, 20))
    gs = mpl.gridspec.GridSpec(4, 3, wspace=0.25, hspace=0.25)
    axarr0 = f.add_subplot(gs[0, :])
    axarr10 = f.add_subplot(gs[1, 0])
    axarr20 = f.add_subplot(gs[2, 0])
    axarr30 = f.add_subplot(gs[3, 0])
    axarr11 = f.add_subplot(gs[1, 1])
    axarr21 = f.add_subplot(gs[2, 1])
    axarr31 = f.add_subplot(gs[3, 1])
    axarr12 = f.add_subplot(gs[1, 2])
    axarr22 = f.add_subplot(gs[2, 2])
    axarr32 = f.add_subplot(gs[3, 2])
    
    # Loss
    ax = axarr0
    ax1 = ax.twinx()

    ax.set_title("Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax1.set_ylabel('Accuracy')

    lns1 = ax.plot(np.arange(epoch + 1), train_loss, color="black", label = 'Training error')
    lns2 = ax.plot(np.arange(epoch + 1), valid_loss, color="gray", linestyle="--", label='Validation error')
    lns3 = ax1.plot(np.arange(epoch + 1), valid_acc, color="blue", linestyle=":", label='Validation accuracy')

    # combine the legend for both axes
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper left")

    # Inputs
    ax10 = axarr10
    ax10.set_title('Inputs')
    ax10.axis('off')
    
    ax20 = axarr20
    ax20.set_title('Inputs')
    ax20.axis('off')
    
    ax30 = axarr30
    ax30.set_title('Inputs')
    ax30.axis('off')

    # Reconstructions
    ax11 = axarr11
    ax11.set_title('Reconstructions')
    ax11.axis('off')
    
    ax21 = axarr21
    ax21.set_title('Reconstructions')
    ax21.axis('off')
    
    ax31 = axarr31
    ax31.set_title('Reconstructions')
    ax31.axis('off')
    
    # Noisy
    ax12 = axarr12
    ax12.set_title('Noisy')
    ax12.axis('off')
    
    ax22 = axarr22
    ax22.set_title('Noisy')
    ax22.axis('off')
    
    ax32 = axarr32
    ax32.set_title('Noisy')
    ax32.axis('off')
    
    #change the number here to display different pictures.
    my_imshow(x[:3],x_hat[:3],x_b[:3],[ax10,ax20,ax30],[ax11,ax21,ax31],[ax12,ax22,ax32])

    if epoch % 24 == 0:
      tmp_result = f'{epoch}_loss_plot.png'
      plt.savefig(tmp_result)

    tmp_img = "tmp_ae_out.png"
    plt.savefig(tmp_img)
    plt.close(f)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)

def my_imshow(imgOriginal,imgReconstructed,imgBlue,o,r,b):
    """ show an image """
    
    for i in range(len(imgOriginal)):
        #----------original images------------------
        img = imgOriginal[i] #/ 2 + 0.5 # unnormalize #<------ WHY THIS??
        npimgOriginal = img.cpu().numpy()

        o[i].imshow(np.transpose(npimgOriginal, (1, 2, 0)))


        #-------------reconstructed images---------
        img = imgReconstructed[i] #/ 2 + 0.5 # unnormalize
        npimgRec = img.cpu().numpy()
        img_out = np.transpose(npimgRec, (1, 2, 0))
        r[i].imshow((img_out * 255).astype(np.uint8))
        
        #------------noisy images--------------
        img = imgBlue[i] #/ 2 + 0.5 # unnormalize
        npimgRec = img.cpu().numpy()
        img_out = np.transpose(npimgRec, (1, 2, 0))
        b[i].imshow((img_out * 255).astype(np.uint8))


# save original and reconstructed images of the selected indices 
def save_img(final_images, lst_idx):
  for idx in lst_idx:
    image = final_images[idx]
    rec_image = final_images[idx+1]
    image_name = f'img{idx}.png'
    rec_image_name = f'rec_img{idx}.png'
    save_image(image, image_name)
    save_image(rec_image, rec_image_name)

# display images from the final test set
def display_multiple_img(final_images, rows = 1, cols=1, start_idx=0):
  images = {}
  idx = 0
  for i in range(rows*cols):
    idx = i + start_idx
    if i % 2 == 0:
      images['Original_'+str(idx)] = final_images[idx]
    else:
      images['Reconstructed_'+str(idx-1)] = final_images[idx]
  figure, ax = plt.subplots(nrows=rows, ncols=cols)
  figure.set_figheight(20)
  figure.set_figwidth(14)    
  for ind, title in enumerate(images):
      img = images[title]
      npimg = img.cpu().numpy()
      img_out = np.transpose(npimg, (1, 2, 0))
      ax.ravel()[ind].imshow((img_out * 255).astype(np.uint8))
      ax.ravel()[ind].set_title(title)
      ax.ravel()[ind].set_axis_off()
  plt.tight_layout()
  plt.show()