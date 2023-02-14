# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import typing
from builtins import *

import canmatrix


class CanCluster(dict):

    def __init__(self, *arg, **kw):

        print("def : cancluster - __init__")

        super(CanCluster, self).__init__(*arg, **kw)
        self._frames = []  # type: typing.List[canmatrix.Frame]
        self._signals = []  # type: typing.List[canmatrix.Signal]
        self._ecus = []  # type: typing.List[canmatrix.Ecu]
        self.update()

    def update_frames(self):  # type: () -> typing.MutableSequence[canmatrix.Frame]

        print("def : cancluster - update_frames")

        frames = []  # type: typing.List[canmatrix.Frame]
        frame_names = []  # type: typing.List[str]
        for matrixName in self:
            for frame in self[matrixName].frames:  # type: canmatrix.Frame
                if frame.name not in frame_names:
                    frame_names.append(frame.name)
                    frames.append(frame)
                else:
                    index = frame_names.index(frame.name)
                    for transmitter in frame.transmitters:
                        frames[index].add_transmitter(transmitter)
                    for receiver in frame.receivers:
                        frames[index].add_receiver(receiver)
        self._frames = frames
        return frames

    def update_signals(self):  # type: () -> typing.MutableSequence[canmatrix.Signal]

        print("def : cancluster - update_signals")

        signals = []  # type: typing.List[canmatrix.Signal]
        signal_names = []  # type: typing.List[str]
        for matrixName in self:
            for frame in self[matrixName].frames:  # type: canmatrix.Frame
                for signal in frame.signals:
                    if signal.name not in signal_names:
                        signal_names.append(signal.name)
                        signals.append(signal)
                    else:
                        index = signal_names.index(signal.name)
                        for receiver in signal.receivers:
                            signals[index].add_receiver(receiver)
        self._signals = signals
        return signals

    def update_ecus(self):  # type: () -> typing.MutableSequence[canmatrix.Ecu]

        print("def : cancluster - update_ecus")

        ecus = []  # type: typing.List[canmatrix.Ecu]
        ecu_names = []  # type: typing.List[str]
        for matrixName in self:
            for ecu in self[matrixName].ecus:  # type: canmatrix.Ecu
                if ecu.name not in ecu_names:
                    ecu_names.append(ecu.name)
                    ecus.append(ecu)
        self._ecus = ecus
        return ecus

    def update(self):

        print("def : cancluster - update")

        self.update_frames()
        self.update_signals()
        self.update_ecus()

    @property
    def ecus(self):  # type: () -> typing.MutableSequence[canmatrix.Ecu]

        print("def : cancluster - ecus")

        if not self._ecus:
            self.update_ecus()
        return self._ecus

    @property
    def frames(self):  # type: () -> typing.MutableSequence[canmatrix.Frame]

        print("def : cancluster - frames")

        if not self._frames:
            self.update_frames()
        return self._frames

    @property
    def signals(self):  # type: () -> typing.MutableSequence[canmatrix.Signal]

        print("def : cancluster - signals")

        if not self._signals:
            self.update_signals()
        return self._signals
