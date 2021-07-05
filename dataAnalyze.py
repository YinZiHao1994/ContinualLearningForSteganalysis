import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import common_utils
import os


def compute_precision_recall_curve(y_true, y_scores):
    return precision_recall_curve(y_true, y_scores)


#
# def show_precision_recall_curve_image(precision, recall, average_precision, title="PR Curve", is_save=False):
#     plt = __create_precision_recall_curve_image(precision, recall, average_precision, title)
#     if is_save:
#         plt.savefig(constant.cache_dir + title + ".png")
#     plt.show()
#
#
# def save_precision_recall_curve_image(precision, recall, average_precision, title="PR Curve", dir_name="PRCurve"):
#     plt = __create_precision_recall_curve_image(precision, recall, average_precision, title)
#     dir_path = constant.cache_dir + dir_name + "/"
#     util.create_dir(dir_path)
#     plt.savefig(dir_path + title + ".png")
#     plt.close()


def __create_precision_recall_curve_image(precision, recall, average_precision, title="PR Curve"):
    plt.figure(title)  # 创建图表1
    plt.title("{}({:.4f})".format(title, average_precision))  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    return plt


def compute_average_precision(y_true, y_scores):
    return average_precision_score(y_true, y_scores)


def save_loss_plot(dir_path, iteration, loss, title, dir_name="lossCurve"):
    dir_path = os.path.join(dir_path, dir_name)
    common_utils.create_dir(dir_path, False)
    plt.figure(title)  # 创建图表1
    plt.title(title)  # give plot a title
    plt.xlabel('iteration')  # make axis labels
    plt.ylabel('loss')
    plt.plot(iteration, loss)
    plt.savefig(os.path.join(dir_path, title + ".png"))
    plt.close()
    # plt.show()


def save_accurate_plot(dir_path, iteration, acc, title, dir_name="accurateCurve"):
    dir_path = os.path.join(dir_path, dir_name)
    common_utils.create_dir(dir_path, False)
    plt.figure(title)  # 创建图表1
    plt.title(title)  # give plot a title
    plt.xlabel('iteration')  # make axis labels
    plt.ylabel('loss')
    plt.plot(iteration, acc)
    plt.savefig(os.path.join(dir_path, title + ".png"))
    plt.close()
    # plt.show()
