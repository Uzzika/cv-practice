import argparse
import NN
import bow
import cv2
from load_data import load_images_from_split

def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', help='Path to train split file', type=str)
    parser.add_argument('-e', '--test_file', help='Path to test split file', type=str)
    parser.add_argument('-m', '--method', help='Method to apply',
                        choices=['BOW', 'NN'],
                        required=True)
    parser.add_argument('-j', '--type', help='Type of job: train or test',
                        choices=['train', 'test', 'draw'],
                        required=True)
    parser.add_argument('--clf', help='Path to classifier for test mode', type=str, default='model.joblib')
    parser.add_argument('--model', help='Path to NN model for test mode', type=str, default='model.keras')
    parser.add_argument('--output', help='Path to save classifier for train mode', type=str, default='model.joblib')
    parser.add_argument('--clusters_name', choices=['KMeans', 'MiniBatch'], help='Name of clusters method', type=str, default='MiniBatch')
    parser.add_argument('--clf_name', choices=['KNN', 'SVC'],help='Nmae of clf method', type=str, default='SVC')
    parser.add_argument('--k_nearest', help='Number of neighbors in KNN', type=int, default=5)
    parser.add_argument('--clusters', help='Number of clusters in clusterization', type=int, default=300)
    parser.add_argument('--batch_size', help='Batch size for KMeansMiniBatch', type=int, default=1000)
    parser.add_argument('--model_name', choices=["VGG"], help='Name of pretrained nn model', type=str, default='VGG')
    parser.add_argument('--draw_img', help='Path to image to draw descriptors on', type=str, default='data\\NNSUDataset\\01_NizhnyNovgorodKremlin\\20250227_151756.jpg')

    return parser.parse_args()

def main():
    args = cli_argument_parser()

    if args.method == "BOW":
        if args.type == "train":
            train_images, train_labels = load_images_from_split(args.train_file)
            bow_ex = bow.BOW(args.clusters_name, args.clf_name, args.k_nearest, args.clusters, args.batch_size)
            bow_ex.sift_descr(train_images)
            bow_ex.bag_of_words()
            bow_ex.bow_histograms()
            bow_ex.train_bow_model(train_labels)
            bow_ex.save_model(args.output)
            print(f"\nDone!")
        if args.type == "test":
            test_images, test_labels = load_images_from_split(args.test_file)
            bow_ex = bow.BOW()
            if args.clf:
                bow_ex.my_load_model(args.clf)
            else:
                raise ValueError("Classifier path (--clf) is required for test mode")
            bow_ex.sift_descr(test_images)
            bow_ex.bow_histograms()
            preds, acc = bow_ex.test_bow_model(test_labels)
            print("Accuracy: ", acc)
        if args.type == "draw":
            img = cv2.imread(args.draw_img)
            bow_ex = bow.BOW()
            bow_ex.print_descr(img)
    if args.method == "NN":
        if args.type == "train":
            train_images, train_labels = load_images_from_split(args.train_file, mode = "NN")
            nn = NN.NN(args.model_name)
            nn.train_nn(train_images, train_labels)
            print("\nDone!")
        if args.type == "test":
            test_images, test_labels = load_images_from_split(args.test_file, mode = "NN")
            nn = NN.NN()
            nn.load(args.model)
            acc,preds = nn.test_nn(test_images,test_labels)
            print("Accuracy: ", acc)
main()
