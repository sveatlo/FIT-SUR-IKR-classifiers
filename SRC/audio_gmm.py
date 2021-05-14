from __future__ import division
import numpy as np
from ikrlib import wav16khz2mfcc, train_gmm, logpdf_gmm

NUM_CLASSES = 31

if __name__ == "__main__":
    M = []
    Ws = []
    MUs = []
    COVs = []

    for i in range(NUM_CLASSES):
        id = i + 1
        print("Loading data for class {}".format(id))
        train = np.vstack(wav16khz2mfcc("train/{}".format(id)).values())

        print("Training model for class {}".format(id))
        M.append(32)
        Ws.append(np.ones(M[i]) / M[i])
        MUs.append(train[np.random.randint(1, len(train), M[i])])
        COVs.append([np.var(train, axis=0)] * M[i])

        n = 15
        for iteration in range(n):
            [Ws[i], MUs[i], COVs[i], TTL] = train_gmm(train, Ws[i], MUs[i], COVs[i])
            print("Training iteration: {}/{}, total log-likelihood: {}".format(iteration + 1, n, TTL))

    errors = 0
    trials = 0
    for i in range(NUM_CLASSES):
        id = i + 1
        test = list(wav16khz2mfcc("dev/{}".format(id)).values())

        for j, test_data in enumerate(test):
            log_lh = []
            for ii in range(NUM_CLASSES):
                log_lh.append(sum(logpdf_gmm(test_data, Ws[ii], MUs[ii], COVs[ii])))
            winning_class_ind = np.argmax(log_lh)
            print("Correct class {} | Winning class {} with value {}".format(i + 1, winning_class_ind + 1, log_lh[winning_class_ind]))
            errors = (errors + 1) if not ((winning_class_ind) == i) else errors
            trials += 1

    print("------------------------------------------")
    print("False predictions: {} out of {}.".format(errors, trials))
    print("Error ratio: {}".format(errors/trials))
    print("------------------------------------------")

    print("Model evaluation...")
    with open("GMM_audio_results", "w") as f:
        eval = wav16khz2mfcc("eval/")
        eval_names = [x.split("\\")[1].split(".")[0] for x in list(eval.keys())]
        eval_vals = list(eval.values())

        for j, eval_data in enumerate(eval_vals):
            log_lh = []
            for ii in range(NUM_CLASSES):
                log_lh.append(sum(logpdf_gmm(eval_data, Ws[ii], MUs[ii], COVs[ii])))
            winning_class_ind = np.argmax(log_lh)

            # logs_lh_string = ' '.join(["%.2f" % number for number in log_lh])
            logs_lh_string = ["NaN" for x in range(NUM_CLASSES)]
            f.write("{} {} {}\n".format(eval_names[j], winning_class_ind + 1, logs_lh_string))