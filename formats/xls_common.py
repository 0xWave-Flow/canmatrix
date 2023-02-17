# -*- coding: utf-8 -*-
# Copyright (c) 2013, Eduard Broecker
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that
# the following conditions are met:
#
#    Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
#    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

from __future__ import absolute_import, division, print_function

import signal
import typing
from builtins import *

import canmatrix




def get_frame_info(db, frame):

    print("def : format - xls_common - get_frame_info")

    # type: (canmatrix.CanMatrix, canmatrix.Frame) -> typing.List[str]
    ret_array = []  # type: typing.List[str]

    if db.type == canmatrix.matrix_class.CAN:
        # frame-id
        if frame.arbitration_id.extended:
            ret_array.append("%3Xxh" % frame.arbitration_id.id)
        else:
            ret_array.append("%3Xh" % frame.arbitration_id.id)
    elif db.type == canmatrix.matrix_class.FLEXRAY:
        ret_array.append("TODO")
    elif db.type == canmatrix.matrix_class.SOMEIP:
        ret_array.append("%3Xh" % frame.header_id)

    # frame-Name
    ret_array.append(frame.name)

    ret_array.append(frame.effective_cycle_time)

    return ret_array


def removesuffix(input_string, suffix):

    #print("def : format - xls_common - removesuffix - 1.{} - 2.{}".format(input_string,suffix))

    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def get_signal(db, frame, sig, motorola_bit_format):

    print("def : format - xls_common - get_signal - [{},{}]".format(frame.name,sig.name))
    # print("def : format - xls_common - get_signal - [{},{},{}]".format(db,frame,sig))

    # type: (canmatrix.CanMatrix, canmatrix.Frame, canmatrix.Signal, str) -> typing.Tuple[typing.List, typing.List]
    front_array = []  # type: typing.List[typing.Union[str, float]]

    # Result
    front_array.append('/')
    # No.
    front_array.append('/')
    try:
        front_array.append(frame.transmitters[0])
    except:
        front_array.append('None')
        # print('No transmitter!')
    try:
        front_array.append(sig.receivers[0])
    except:
        front_array.append('None')
        # print('No receiver!')

    # signal name
    front_array.append(sig.name)

    # ASIL level
    front_array.append('/')
    #front_array.append(sig.)
    # Signal Description
    front_array.append('/')
    # Used by function
    front_array.append('/')

    # frame-Name
    front_array.append(frame.name)

    # frame-ID
    if db.type == canmatrix.matrix_class.CAN:
        # frame-id
        if frame.arbitration_id.extended:
            front_array.append("0x%3X" % frame.arbitration_id.id)
        else:
            front_array.append("0x%3X" % frame.arbitration_id.id)
    elif db.type == canmatrix.matrix_class.FLEXRAY:
        front_array.append("TODO")
    elif db.type == canmatrix.matrix_class.SOMEIP:
        front_array.append("0x%3X" % frame.header_id)

    # Message Type
    front_array.append('/')
    # Signal Type
    front_array.append('/')
    # Detailed description of E/M Message sending behavior Tx
    front_array.append('/')

    # Detailed description of E/M Message sending behavior Rx
    front_array.append('/')

    # cycle time
    front_array.append(frame.cycle_time)

    # Signal latency budget Tx(ms)
    front_array.append('/')

    # Signal latency budget Rx(ms)
    front_array.append('/')

    # DLC
    front_array.append(int(frame.size))

    # MSB
    front_array.append(sig.msb)

    # LSB
    temp = sig.msb - (sig.msb % 8) + 7 - (sig.msb % 8) + sig.size - 1
    # startBitInternal = startBitInternal + self.size - 1
    lsb = temp - (temp % 8) + 7 - (temp % 8)
    front_array.append(lsb)

    # signal size
    front_array.append(sig.size)

    # eval byteorder (little_endian: intel == True / motorola == 0)
    if sig.is_little_endian:
        front_array.append("intel")
    else:
        front_array.append("motorola")

    # data type
    if not sig.is_signed:
        front_array.append('Unsigned')
    else:
        if sig.size == 32:
            front_array.append('IEEE Float')
        elif sig.size == 62:
            front_array.append('IEEE Double')
        else:
            front_array.append('Signed')
    # First valid signal duration Tx(ms)
    front_array.append('/')

    # First valid signal duration Rx(ms)
    front_array.append('/')

    # start-value of signal available
    front_array.append(sig.initial_value)
    # Alternative value
    front_array.append('/')

    # factor
    front_array.append(float(sig.factor))
    # Measured Resolution Tx
    front_array.append('/')

    # Measured Resolution Rx
    front_array.append('/')

    # Accuracy Tx
    front_array.append('/')

    # offset
    front_array.append(float(sig.offset))

    # p-min
    front_array.append(float(sig.min))

    # p-max
    front_array.append(float(sig.max))

    # unit
    front_array.append(sig.unit)
    # Coding
    coding = str()
    for val in sorted(sig.values.keys()):
        coding += str(val) + ':' + sig.values[val] + '\n'
    #coding = coding.removesuffix('\n')
    coding = removesuffix(coding,'\n')
    front_array.append(coding)

    # Signal Group
    front_array.append('/')

    # Message Segment
    front_array.append('/')

    # Variant and options
    front_array.append('/')

    # eval comment:
    comment = sig.comment if sig.comment else ""

    # write comment and size of signal in sheet
    front_array.append(comment)
    # Receiver ECU list

    #print("def : format - xls_common - get_signal - {}".format(front_array))

    return front_array
