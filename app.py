import os, glob
import cv2 as cv
from absl import app, logging,flags

FLAGS = flags.FLAGS

flags.DEFINE_string( 'image_dir', './images', 'Path to image directory')
flags.DEFINE_string('output_dir', './results', 'Path to output directory')
flags.DEFINE_integer('downsample_size', 256, 'Size of downsampled image', lower_bound=0)
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

flags.mark_flag_as_required('image_dir')

def read_images(img_path) :
    imgs = {}
    for img in glob.glob(os.path.join(img_path,"*.*")):
        try:
            imgs[img]=cv.imread(img)
        except:
            logging.error(f'error in reading {img}!')
    return imgs

def down_sample(imgs, down_sample_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_path, img in imgs.items():
        out_path = os.path.join(output_dir ,os.path.basename(img_path))
        if not os.path.exists(out_path) :
            down_sampled_img = cv.resize(img,(down_sample_size, down_sample_size))
            cv.imwrite(out_path, down_sampled_img)
        else:
            logging.warning(f'file {out_path} exists!')
    return 0


def main(argv):
    del argv
    logging.info('Starting down sampling!')
    imgs = read_images(FLAGS.image_dir)
    down_sample(imgs, FLAGS.downsample_size, FLAGS.output_dir)
    logging.info('Down sampling finished!')

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)

