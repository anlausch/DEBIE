import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import codecs

def plot_loss(path_to_log="./output/data/weat_8_postspec_4_wo_original_adv/drp=0.9_rf=0.4_adv=True/drp=0.9_rf=0.4_adv=True.log", output_path="./output/data/weat_8_postspec_4_wo_original_adv/drp=0.9_rf=0.4_adv=True/loss.png"):
  """
  :param output_path:
  :param task:
  :return:
   >>> plot_loss()
  """
  generator_losses = []
  discriminator_losses = []
  with codecs.open(path_to_log, "r", "utf8") as f:
    for line in f.readlines():
      if len(line.split(", Generator loss: ")) > 1:
        generator_losses.append(float(line.split(", Generator loss: ")[1].split(", Discriminator loss: ")[0]))
        discriminator_losses.append(float(line.split(", Discriminator loss: ")[1]))


  sns.set()
  with sns.plotting_context("paper"):
    ind = range(len(discriminator_losses))

    fig, ax = plt.subplots()
    ax.plot(ind, generator_losses)
    ax.plot(ind, discriminator_losses)
    plt.legend(['generator', 'discriminator'], loc='upper left')

    ax.grid()

    fig.savefig(output_path)
    #plt.show()
    print("Done")
