from loguru import logger


def load_best_weights(model, path, name='model*.h5', best="min"):
    """
    Load the best weight file in a given directory
    :param model: tf model
    :param path: a posix path
    :param name: glob name to look for
    """
    first_or_last = 0 if best == "min" else -1
    best_weights_file = sorted(path.glob(name), key=lambda x: x.name.split('_')[-1][:-3])[first_or_last]
    model.load_weights(best_weights_file, by_name=True)
    logger.info(f"Weights from checkpoint {best_weights_file} restored.")
