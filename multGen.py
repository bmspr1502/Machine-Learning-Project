import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.cm as cm
import matplotlib.mlab as mlab 
from collections import namedtuple


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('trained_models', 'model-29'))
parser.add_argument('--text', dest='text', type=str, default=None)
parser.add_argument('--style', dest='style', type=int, default=0)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--info', dest='info', action='store_true', default=False)
parser.add_argument('--output', dest="output", type=str, default='testSample.png')
args = parser.parse_args()


def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


def sample_text(sess, args_text, translation, style=None):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    vs = namedtuple('Params', fields)(
        *[tf.compat.v1.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0] # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text] # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len

        print('\r[{:5d}] sampling... {}'.format(s, 'priming' if is_priming else 'synthesis'), end='')

        e, pi, mu1, mu2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                               vs.std1, vs.std2, vs.rho, vs.finish,
                                               vs.phi, vs.window, vs.kappa],
                                              feed_dict={
                                                  vs.coordinates: coord[None, None, ...],
                                                  vs.sequence: sequence_prime if is_priming else sequence,
                                                  vs.bias: args.bias
                                              })

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                           std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not args.force and finish[0, 0] > 0.8:
                print('\nFinished sampling!\n')
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords


def gen(text, w_style, w_bias, info, output):
    my_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''
    info = False
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph(args.model_path + '.meta')
        saver.restore(sess, args.model_path)

        """
        while True:
            
            if args.text is not None:
                args_text = args.text
            else:
                args_text = input('What to generate: ')
        """
        style = None
        w_style = int(w_style)
        args.bias = w_bias
        if w_style != 404:
            args.style = w_style
        if args.style is not None:
            style = None
            with open(os.path.join('data', 'styles.pkl'), 'rb') as file:
                styles = pickle.load(file)

            if args.style > len(styles[0]):
                raise ValueError('Requested style is not in style list')

            style = [styles[0][args.style], styles[1][args.style]]

        text_file = text

        textinput = []

        for line in text_file:
            textinput.append(line)

        nooflines = len(textinput)
        i = 0

        fig,ax = plt.subplots(nooflines)

        for line in textinput:
            args_text= line
            phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, args_text, translation, style)

            strokes = np.array(stroke_data)
            epsilon = 1e-8
            strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)

            for stroke in split_strokes(cumsum(np.array(coords))):
                ax[i].plot(stroke[:, 0], -stroke[:, 1], color="black")
            ax[i].set_title(args_text)
            ax[i].set_aspect('equal')
            i+=1
                
            
        
        print("Saving image to: ", os.path.join(my_path,'images','multiLine' , output))
        plt.savefig(os.path.join(my_path,'images','multiLine', output))
        plt.show()

        """
        if args.text is not None:
            break
        """



def main():
    text = open(args.text)

    style = args.style
    bias = args.bias
    info = args.info
    output = args.output

    print(args)
    gen(text, style, bias, info, output)
    text.close()

if __name__ == '__main__':
    main()