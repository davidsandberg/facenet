from argparse import ArgumentParser, FileType, Namespace

from evaluator.evaluator import Evaluator
from metrics.metrics import DistanceMetric, ThresholdMetric


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        required=True,
                        help='Path to the image directory.')
    parser.add_argument('--pairs_fname',
                        type=FileType('r', encoding='utf-8'),
                        required=True,
                        help='Filename of pairs.txt')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to the facial recognition model')
    parser.add_argument(
        '--is_insightface',
        action='store_true',
        help='Set this flag if the model is insightface')
    distance_metrics = [str(metric)
                        .replace(f'{DistanceMetric.__qualname__}.', '')
                        for metric in DistanceMetric]
    parser.add_argument(
        '--distance_metric',
        type=str,
        required=True,
        choices=distance_metrics,
        help=f"Distance metric for face verification: {distance_metrics}.")
    parser.add_argument('--threshold_start',
                        type=float,
                        required=True,
                        help='Start value for distance threshold.')
    parser.add_argument('--threshold_end',
                        type=float,
                        required=True,
                        help='End value for distance threshold')
    parser.add_argument(
        '--threshold_step',
        type=float,
        required=True,
        help='Step size for iterating in cross validation search.')
    threshold_metrics = [str(metric)
                         .replace(f'{ThresholdMetric.__qualname__}.', '')
                         for metric in ThresholdMetric]
    parser.add_argument('--threshold_metric',
                        type=str,
                        required=True,
                        choices=threshold_metrics,
                        help='metric for calculating optimal threshold.')
    parser.add_argument(
        '--embedding_size',
        type=int,
        required=True,
        help='Size of face vectors from face_verification_container.')
    parser.add_argument(
        '--remove_empty_embeddings_flag',
        action='store_true',
        help='Instead of a default encoding for images where\
faces are not detected, remove them')
    parser.add_argument(
        '--prealigned_flag',
        action='store_true',
        help='Specify if the images have already been aligned.')
    return parser.parse_args()


def _main(args: Namespace) -> None:
    evaluator = Evaluator.create_evaluator(args)
    evaluation_results = evaluator.evaluate()
    print('Evaluation results: ', evaluation_results)
    parser_metrics = evaluator.compute_metrics()
    print('Parser metrics: ', parser_metrics)


def _cli() -> None:
    args = _parse_arguments()
    args.pairs_fname.close()
    args.pairs_fname = args.pairs_fname.name
    _main(args)


if __name__ == '__main__':
    _cli()
