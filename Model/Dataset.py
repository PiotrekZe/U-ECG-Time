import numpy as np
import wfdb
import os


class Dataset:
    def __init__(self, path, length=1800, aug_data=False):
        self.path = path
        self.length = length
        self.aug_data = aug_data  # to do: data augmentation

    def __segmentate_data(self, signal, peaks, labels):
        start, end = 0, self.length
        signals_tab, peaks_tab, labels_tab, peaks_idx = [], [], [], []
        while end < signal.shape[0] and start < signal.shape[0]:
            signals_tab.append(signal[start:end])
            peaks_tab.append(peaks[start:end])
            labels_tab.append(labels[start:end])

            idx = np.where(peaks_tab[-1] == 1)[0]
            if len(idx) == 0:
                tmp = 0
            else:
                tmp = idx[-1]
            peaks_idx.append(tmp)
            if start == (start + tmp):
                start = start + self.length
                end = start + self.length
            else:
                start += tmp
                end = start + self.length
        return (
            np.array(signals_tab),
            np.array(peaks_tab),
            np.array(labels_tab),
            peaks_idx,
        )

    def __read_file(self, file):
        file_path = os.path.join(self.path, file)

        record = wfdb.rdrecord(file_path)
        annotation = wfdb.rdann(file_path, "atr")

        size = record.p_signal.shape[0]

        file_peaks = np.zeros(size)  # coords of R-peaks
        file_labels = np.zeros(size)  # labels of each time step
        tmp_sample, tmp_symbol = [], []  # coors and symbols of R-peaks

        # get rid of noninformative samples
        for i in range(len(annotation.symbol)):
            # idk if that list is the same for other data
            if annotation.symbol[i] not in ["+", "~", "x", "|"]:
                tmp_sample.append(annotation.sample[i])
                tmp_symbol.append(annotation.symbol[i])
                file_peaks[annotation.sample[i]] = 1

        # creating labels list based on the
        for i in range(len(tmp_symbol) - 1):
            if tmp_symbol[i] in ["N", "L", "R", "e", "j"]:
                file_labels[tmp_sample[i] : tmp_sample[i + 1]] = 0
            elif tmp_symbol[i] in ["A", "a", "J", "S"]:
                file_labels[tmp_sample[i] : tmp_sample[i + 1]] = 1
            elif tmp_symbol[i] in ["V", "E"]:
                file_labels[tmp_sample[i] : tmp_sample[i + 1]] = 2
            elif tmp_symbol[i] in ["F"]:
                file_labels[tmp_sample[i] : tmp_sample[i + 1]] = 3
            elif tmp_symbol[i] in ["/", "f", "Q"]:
                file_labels[tmp_sample[i] : tmp_sample[i + 1]] = 4

        # get rid of time steps behind first classified R-peak, and after the last one
        file_signals = record.p_signal[tmp_sample[0] : tmp_sample[-1]]
        file_labels = file_labels[tmp_sample[0] : tmp_sample[-1]]
        file_peaks = file_peaks[tmp_sample[0] : tmp_sample[-1]]

        # segmentate signals, labels, peaks into n sec segments
        segmented_signals, segmented_peaks, segmented_outputs, peak_idx = (
            self.__segmentate_data(file_signals, file_peaks, file_labels)
        )

        return segmented_signals, segmented_peaks, segmented_outputs, peak_idx

    def read_dataset(self):
        f = open(f"{self.path}/RECORDS", "r")
        files = f.read().replace("\n", " ").split()

        peaks, targets, inputs, peaks_idx = [], [], [], []

        for file in files:
            segmented_signals, segmented_peaks, segmented_outputs, peak_idx = (
                self.__read_file(file)
            )

            peaks.extend(segmented_peaks)
            inputs.extend(segmented_signals)
            targets.extend(segmented_outputs)
            peaks_idx.extend(peak_idx)

        return np.array(inputs), np.array(targets), np.array(peaks), np.array(peaks_idx)
