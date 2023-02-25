# -*- coding: utf-8 -*-
# Copyright (c) 2013, Eduard Broecker
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that
# the following conditions are met:
#
#    Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#    Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,   PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR  OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# TODO: Definitions should be disassembled

from __future__ import absolute_import, division, print_function

import decimal
import fnmatch
import itertools
import logging
import math
import struct
import typing
import warnings
from builtins import *

import attr
from past.builtins import basestring
from six.moves import zip_longest

import canmatrix.copy
import canmatrix.types
import canmatrix.utils

if attr.__version__ < '17.4.0':  # type: ignore
    raise RuntimeError("need attrs >= 17.4.0")
logger = logging.getLogger(__name__)
defaultFloatFactory = decimal.Decimal  # type: typing.Callable[[typing.Any], canmatrix.types.PhysicalValue]


class ExceptionTemplate(Exception):
    def __call__(self, *args):
        return self.__class__(*(self.args + args))


class StartbitLowerZero(ExceptionTemplate): pass


class EncodingComplexMultiplexed(ExceptionTemplate): pass


class MissingMuxSignal(ExceptionTemplate): pass


class DecodingComplexMultiplexed(ExceptionTemplate): pass


class DecodingFrameLength(ExceptionTemplate): pass


class ArbitrationIdOutOfRange(ExceptionTemplate): pass


class J1939needsExtendedIdetifier(ExceptionTemplate): pass


class DecodingConatainerPdu(ExceptionTemplate): pass


class EncodingConatainerPdu(ExceptionTemplate): pass


def arbitration_id_converter(source):  # type: (typing.Union[int, ArbitrationId]) -> ArbitrationId

    # print("def : canmatrix - arbitration_id_converter")

    """Converter for attrs which accepts ArbitrationId itself or int."""
    # print(isinstance(source,ArbitrationId))
    # print(type(ArbitrationId.from_compound_integer(source)))
    return source if isinstance(source, ArbitrationId) else ArbitrationId.from_compound_integer(source)


@attr.s
class Ecu(object):
    """
    Represents one ECU.
    """

    name = attr.ib()  # type: str
    comment = attr.ib(default=None)  # type: typing.Optional[str]
    attributes = attr.ib(factory=dict, repr=False)  # type: typing.MutableMapping[str, typing.Any]

    def attribute(self, attribute_name, db=None, default=None):  # type: (str, CanMatrix, typing.Any) -> typing.Any

        print("def : canmatrix - Ecu - attribute")

        """Get Board unit attribute by its name.

        :param str attribute_name: attribute name.
        :param CanMatrix db: Optional database parameter to get global default attribute value.
        :param default: Default value if attribute doesn't exist.
        :return: Return the attribute value if found, else `default` or None
        """
        if attribute_name in self.attributes:
            return self.attributes[attribute_name]
        elif db is not None:
            if attribute_name in db.ecu_defines:
                define = db.ecu_defines[attribute_name]
                return define.defaultValue
        return default

    def add_attribute(self, attribute, value):  # type (attribute: str, value: typing.Any) -> None

        print("def : canmatrix - Ecu - add_attribute")

        """
        Add the Attribute to current ECU. If the attribute already exists, update the value.

        :param str attribute: Attribute name
        :param any value: Attribute value
        """
        self.attributes[attribute] = value

    def del_attribute(self, attribute):

        print("def : canmatrix - Ecu - del_attribute")

        if attribute in self.attributes:
            del self.attributes[attribute]

    def add_comment(self, comment):  # type: (str) -> None

        print("def : canmatrix - Ecu - add_comment")

        """
        Set ECU comment.

        :param str comment: BU comment/description.
        """
        self.comment = comment


def normalize_value_table(table):  # type: (typing.Mapping) -> typing.MutableMapping[int, typing.Any]

    # print("def : canmatrix - normalize_value_table - [{}]".format(table))

    return {int(k): v for k, v in table.items()}


@attr.s(eq=False)
class Signal(object):
    """
    Represents a Signal in CAN Matrix.

    Signal has following attributes:

    * name
    * start_bit (internal start_bit, see get/set_startbit also)
    * size (in Bits)
    * is_little_endian (1: Intel, 0: Motorola)
    * is_signed (bool)
    * factor, offset, min, max
    * receivers  (ECU Name)
    * attributes, _values, unit, comment
    * _multiplex ('Multiplexor' or Number of Multiplex)
    """

    name = attr.ib(default="")  # type: str
    # float_factory = attr.ib(default=defaultFloatFactory)
    float_factory = defaultFloatFactory  # type: typing.Callable[[typing.Any], canmatrix.types.PhysicalValue]
    # start_bit = attr.ib(default=0)  # type: int
    msb = attr.ib(default=0)
    lsb = attr.ib(default=0)
    size = attr.ib(default=0)  # type: int
    is_little_endian = attr.ib(default=True)  # type: bool
    is_signed = attr.ib(default=True)  # type: bool
    offset = attr.ib(converter=float_factory, default=float_factory(0.0))  # type: canmatrix.types.PhysicalValue
    factor = attr.ib(converter=float_factory, default=float_factory(1.0))  # type: canmatrix.types.PhysicalValue

    unit = attr.ib(default="")  # type: str
    receivers = attr.ib(factory=list)  # type: typing.MutableSequence[str]
    comment = attr.ib(default=None)  # type: typing.Optional[str]
    multiplex = attr.ib(default=None)  # type: typing.Union[str, int]

    mux_value = attr.ib(default=None)
    is_float = attr.ib(default=False)  # type: bool
    is_ascii = attr.ib(default=False)  # type: bool
    type_label = attr.ib(default="")
    enumeration = attr.ib(default=None)  # type: typing.Optional[str]
    comments = attr.ib(factory=dict)  # type: typing.MutableMapping[int, str]
    attributes = attr.ib(factory=dict)  # type: typing.MutableMapping[str, typing.Any]
    values = attr.ib(converter=normalize_value_table, factory=dict)  # type: typing.MutableMapping[int, str]
    mux_val_grp = attr.ib(factory=list)  # type: typing.MutableSequence[list]
    muxer_for_signal = attr.ib(default=None)  # type: typing.Optional[str]

    # offset = attr.ib(converter=float_factory, default=0.0)  # type: float # ??
    calc_min_for_none = attr.ib(default=True)  # type: bool
    calc_max_for_none = attr.ib(default=True)  # type: bool

    cycle_time = attr.ib(default=0)  # type: int
    initial_value = attr.ib(converter=float_factory, default=float_factory(0.0))  # type: canmatrix.types.PhysicalValue
    frames = attr.ib(default=list)  # type: typing.MutableSequence[Frame]

    min = attr.ib(
        converter=lambda value, float_factory=float_factory: (
            float_factory(value)
            if value is not None
            else value
        )
    )  # type: typing.Union[int, decimal.Decimal, None]

    @min.default
    def set_default_min(self):  # type: () -> canmatrix.types.OptionalPhysicalValue

        print("def : canmatrix - Signal - set_default_min")

        return self.set_min()

    max = attr.ib(
        converter=lambda value, float_factory=float_factory: (
            float_factory(value)
            if value is not None
            else value
        )
    )  # type: canmatrix.types.OptionalPhysicalValue

    @max.default
    def set_default_max(self):

        print("def : canmatrix - Signal - set_default_max")

        return self.set_max()

    def __attrs_post_init__(self):

        # print("def : canmatrix - Signal - __attrs_post_init__")

        self.multiplex = self.multiplex_setter(self.multiplex)

    def add_frame(self, frame):

        # print("def : canmatrix - Signal - add_frame - [{}]".format(frame))

        # type: (Frame) -> Frame
        """
        Add Frame to Signal.

        :param frame:
        :return: the frame added.
        """
        self.frames.append(frame)

        return self.frames[len(self.frames) - 1]

    @property
    def spn(self):  # type: () -> typing.Optional[int]

        print("def : canmatrix - Signal - spn")

        """Get signal J1939 SPN or None if not defined.

        :rtype: typing.Optional[int]"""
        return self.attributes.get("SPN", None)

    def multiplex_setter(self, value):

        # print("def : canmatrix - Signal - multiplex_setter")

        self.mux_val = None
        self.is_multiplexer = False
        ret_multiplex = None
        if value is not None and value != 'Multiplexor':
            ret_multiplex = int(value)
            self.mux_val = int(value)
        elif value == 'Multiplexor':
            self.is_multiplexer = True
            ret_multiplex = value
        return ret_multiplex

    def multiplexer_value_in_range(self, mux_value):

        print("def : canmatrix - Signal - multiplexer_value_in_range")

        if len(self.mux_val_grp) > 0 and mux_value is not None:
            for mux_min, mux_max in self.mux_val_grp:
                if mux_value >= mux_min and mux_value <= mux_max:
                    return True
            else:
                return False
        else:
            return mux_value == self.mux_val

    def attribute(self, attributeName, db=None, default=None):

        print("def : canmatrix - Signal - attribute - [{}]".format(self.name,attributeName))

        # type: (str, CanMatrix, typing.Any) -> typing.Any
        """Get any Signal attribute by its name.

        :param str attributeName: attribute name, can be mandatory (ex: start_bit, size) or optional (customer) attribute.
        :param CanMatrix db: Optional database parameter to get global default attribute value.
        :param default: Default value if attribute doesn't exist.
        :return: Return the attribute value if found, else `default` or None
        """
        if attributeName in attr.fields_dict(type(self)):
            return getattr(self, attributeName)
        if attributeName in self.attributes:
            return self.attributes[attributeName]
        if db is not None:
            if attributeName in db.signal_defines:
                define = db.signal_defines[attributeName]
                return define.defaultValue
        return default

    def add_comment(self, comment):

        print("def : canmatrix - Signal - add_comment - {}".format(comment))

        """
        Set signal description.

        :param str comment: description
        """

        #comment = "SAMPLE_COMMENT"
        self.comment = comment

    def add_receiver(self, receiver):

        print("def : canmatrix - Signal - add_receiver")

        """Add signal receiver (ECU).

        :param str receiver: ECU name.
        """
        if receiver not in self.receivers:
            self.receivers.append(receiver)

    def del_receiver(self, receiver):

        print("def : canmatrix - Signal - del_receiver")

        """
        Remove receiver (ECU) from signal

        :param str receiver: ECU name.
        """
        if receiver in self.receivers:
            self.receivers.remove(receiver)

    def add_attribute(self, attribute, value):

        print("def : canmatrix - Signal - add_attribute")

        """
        Add user defined attribute to the Signal. Update the value if the attribute already exists.

        :param str attribute: attribute name
        :param value: attribute value
        """
        self.attributes[attribute] = value

    def del_attribute(self, attribute):

        print("def : canmatrix - Signal - del_attributess")

        """
        Remove user defined attribute from Signal.

        :param str attribute: attribute name
        """
        if attribute in self.attributes:
            del self.attributes[attribute]

    def add_values(self, value, valueName):

        print("def : canmatrix - Signal - add_values - [{},{}]".format(value,valueName))

        """
        Add named Value Description to the Signal.

        :param int or str value: signal value (0xFF)
        :param str valueName: Human readable value description ("Init")
        """
        if isinstance(value, defaultFloatFactory):
            self.values[value.to_integral()] = valueName
        else:
            self.values[int(str(value), 0)] = valueName

    def calculate_raw_range(self):

        print("def : canmatrix - Signal - calculate_raw_range")

        """Compute raw signal range based on Signal bit width and whether the Signal is signed or not.

        :return: Signal range, i.e. (0, 15) for unsigned 4 bit Signal or (-8, 7) for signed one.
        :rtype: tuple
        """
        factory = (
            self.float_factory
            if self.is_float
            else int
        )
        size_to_calc = self.size if self.size <= 128 else 128
        if size_to_calc != self.size:
            logger.info("max calculation for {} not possible using 128 as base for max value".format(self.size))
        rawRange = 2 ** (size_to_calc - (1 if self.is_signed else 0))

        return (
            factory(-rawRange if self.is_signed else 0),
            factory(rawRange - 1),
        )

    def set_min(self, min=None):

        print("def : canmatrix - Signal - set_min")

        # type: (canmatrix.types.OptionalPhysicalValue) -> canmatrix.types.OptionalPhysicalValue
        """Set minimal physical Signal value.

        :param min: minimal physical value. If None and enabled (`calc_min_for_none`), compute using `calc_min`
        """
        self.min = min
        if self.calc_min_for_none and self.min is None:
            self.min = self.calc_min()

        return self.min

    def calc_min(self):  # type: () -> canmatrix.types.PhysicalValue

        print("def : canmatrix - Signal - calc_min")

        """Compute minimal physical Signal value based on offset and factor and `calculate_raw_range`."""
        rawMin = self.calculate_raw_range()[0]

        return self.offset + (self.float_factory(rawMin) * self.factor)

    def set_max(self, max=None):

        print("def : canmatrix - Signal - set_max")

        # type: (canmatrix.types.OptionalPhysicalValue) -> canmatrix.types.OptionalPhysicalValue
        """Set maximal signal value.

        :param max: minimal physical value. If None and enabled (`calc_max_for_none`), compute using `calc_max`
        """
        self.max = max

        if self.calc_max_for_none and self.max is None:
            self.max = self.calc_max()

        return self.max

    def calc_max(self):  # type: () -> canmatrix.types.PhysicalValue

        print("def : canmatrix - Signal - calc_max")

        """Compute maximal physical Signal value based on offset, factor and `calculate_raw_range`."""
        rawMax = self.calculate_raw_range()[1]

        return self.offset + (self.float_factory(rawMax) * self.factor)

    def phys2raw(self, value=None):

        #print("def : canmatrix - Signal - phys2raw")

        # type: (canmatrix.types.OptionalPhysicalValue) -> canmatrix.types.RawValue
        """Return the raw value (= as is on CAN).

        :param value: (scaled) value compatible with `decimal` or value choice to encode
        :return: raw unscaled value as it appears on the bus
        :rtype: int or decimal.Decimal
        """
        if value is None:
            value = self.initial_value
            if not (self.min <= value <= self.max):
                value = self.min
        if isinstance(value, basestring):
            for value_key, value_string in self.values.items():
                if value_string == value:
                    value = value_key
                    break
            else:
                raise ValueError(
                    "{} is invalid value choice for {}".format(value, self)
                )

        if not (self.min <= value <= self.max):
            logger.info(
                "Value {} is not valid for {}. Min={} and Max={}".format(
                    value, self, self.min, self.max)
            )

        raw_value = (value - self.offset) / self.factor

        if not self.is_float:
            raw_value = int(raw_value)

        print("def : canmatrix - Signal - phys2raw - {},{}".format(self.name,raw_value))
        return raw_value

    def raw2phys(self, value, decode_to_str=False):

        print("def : canmatrix - Signal - raw2phys")

        # type: (canmatrix.types.RawValue, bool) -> typing.Union[canmatrix.types.PhysicalValue, str]
        """Decode the given raw value (= as is on CAN).

        :param value: raw value compatible with `decimal`.
        :param bool decode_to_str: If True, try to get value representation as *string* ('Init' etc.)
        :return: physical value (scaled)
        """
        if self.is_float:
            value = self.float_factory(value)
        result = value * self.factor + self.offset  # type: typing.Union[canmatrix.types.PhysicalValue, str]
        if decode_to_str:
            for value_key, value_string in self.values.items():
                if value_key == result:
                    result = value_string
                    break
        return result

    def __str__(self):  # type: () -> str

        # print("def : canmatrix - Signal - __str__")

        return self.name


@attr.s(eq=False)
class SignalGroup(object):
    """
    Represents signal-group, containing multiple Signals.
    """
    name = attr.ib()  # type: str
    id = attr.ib()  # type: int
    signals = attr.ib(factory=list, repr=False)  # type: typing.MutableSequence[Signal]
    e2e_trans = attr.ib(default=None)

    def add_signal(self, signal):  # type: (Signal) -> None

        print("def : canmatrix - SignalGroup - add_signal")

        """Add a Signal to SignalGroup.

        :param Signal signal: signal to add
        """
        if signal not in self.signals:
            self.signals.append(signal)

    def del_signal(self, signal):  # type: (Signal) -> None

        print("def : canmatrix - SignalGroup - del_signal")

        """Remove Signal from SignalGroup.

        :param Signal signal: signal to remove
        """
        if signal in self.signals:
            self.signals.remove(signal)

    def by_name(self, name):  # type: (str) -> typing.Union[Signal, None]

        print("def : canmatrix - SignalGroup - by_name - ",self.signals)

        """
        Find a Signal in the group by Signal name.

        :param str name: Signal name to find
        :return: signal contained in the group identified by name
        :rtype: Signal
        """
        for test in self.signals:
            print("def : canmatrix - SignalGroup - by_name - SCAN : {}".format(test))
            if test.name == name:
                return test
        return None

    def __iter__(self):  # type: () -> typing.Iterable[Signal]

        print("def : canmatrix - SignalGroup - __iter__")

        """Iterate over all contained signals."""
        return iter(self.signals)

    def __getitem__(self, name):  # type: (str) -> Signal

        print("def : canmatrix - SignalGroup - __getitem__")

        signal = self.by_name(name)
        if signal:
            return signal
        raise KeyError("Signal '{}' doesn't exist".format(name))


@attr.s
class DecodedSignal(object):
    """
    Contains a decoded signal (frame decoding)

    * rawValue : rawValue (value on the bus)
    * physValue: physical Value (the scaled value)
    * namedValue: value of Valuetable
    * signal: pointer signal (object) which was decoded
    """
    raw_value = attr.ib()  # type: canmatrix.types.RawValue
    signal = attr.ib()  # type: Signal

    @property
    def phys_value(self):  # type: () -> canmatrix.types.PhysicalValue

        print("def : canmatrix - DecodedSignal - phys_value")

        """
        :return: physical Value (the scaled value)
        :rtype: typing.Union[int, decimal.Decimal]
        """
        return self.signal.raw2phys(self.raw_value)

    @property
    def named_value(self):

        print("def : canmatrix - DecodedSignal - name_value")

        """
        :return: value of Valuetable
        :rtype: typing.Union[str, int, decimal.Decimal]
        """
        return self.signal.raw2phys(self.raw_value, decode_to_str=True)


# https://docs.python.org/3/library/itertools.html
def grouper(iterable, n, fillvalue=None):

    print("def : canmatrix - grouper")

    """Collect data into fixed-length chunks or blocks."""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def unpack_bitstring(length, is_float, is_signed, bits):

    print("def : canmatrix - unpack_bitstring")

    # type: (int, bool, bool, typing.Any) -> typing.Union[float, int]
    """
    returns a value calculated from bits
    :param length: length of signal in bits
    :param is_float: value is float
    :param bits: value as bits (array/iterable)
    :param is_signed: value is signed
    :return:
    """

    if is_float:
        types = {
            32: '>f',
            64: '>d'
        }

        float_type = types[length]
        value, = struct.unpack(float_type, bytearray(int(''.join(b), 2) for b in grouper(bits, 8)))
    else:
        value = int(bits, 2)

        if is_signed and bits[0] == '1':
            value -= (1 << len(bits))

    return value


def pack_bitstring(length, is_float, value, signed):

    print("def : canmatrix - pack_bitstring")

    """
    returns a value in bits
    :param length: length of signal in bits
    :param is_float: value is float
    :param value: value to encode
    :param signed: value is signed
    :return:
    """
    if is_float:
        types = {
            32: '>f',
            64: '>d'
        }

        float_type = types[length]
        x = bytearray(struct.pack(float_type, value))
        bitstring = ''.join('{:08b}'.format(b) for b in x)
    else:
        b = '{:0{}b}'.format(int((2 << length) + value), length)
        bitstring = b[-length:]

    return bitstring


@attr.s
class ArbitrationId(object):
    standard_id_mask = ((1 << 11) - 1)
    extended_id_mask = ((1 << 29) - 1)
    compound_extended_mask = (1 << 31)

    id = attr.ib(default=None)
    extended = attr.ib(default=False)  # type: bool

    def __attrs_post_init__(self):

        # print("def : canmatrix - ArbitrationId - __attrs_post_init__")

        if self.extended is None:
            # Mimicking old behaviour for now -- remove in the future
            self.extended = True
            warnings.warn(
                "Please set 'extended' attribute as a boolean instead of "
                "None when creating an instance of ArbitrationId class",
                DeprecationWarning
            )
        if self.extended:
            mask = self.extended_id_mask
        else:
            mask = self.standard_id_mask

        # if self.id != self.id & mask:
        #    raise ArbitrationIdOutOfRange('ID out of range')

    @property
    def j1939_pgn(self):

        print("def : canmatrix - ArbitrationId - j1939_pgn")

        return self.pgn

    @property
    def pgn(self):

        print("def : canmatrix - ArbitrationId - pgn")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        # PGN is bits 8-25 of the 29-Bit Extended CAN-ID
        # Made up of PDU-S (8-15), PDU-F (16-23), Data Page (24) & Extended Data Page (25)
        # If PDU-F >= 240 the PDU-S is interpreted as Group Extension
        # If PDU-F < 240 the PDU-S is interpreted as a Destination Address
        _pgn = 0
        if self.j1939_pdu_format == 2:
            _pgn += self.j1939_ps
        _pgn += self.j1939_pf << 8
        _pgn += self.j1939_dp << 16
        _pgn += self.j1939_edp << 17

        return _pgn

    @pgn.setter
    def pgn(self, value):  # type: (int) -> None

        print("def : canmatrix - ArbitrationId - pgn")

        self.extended = True
        _pgn = value & 0x3FFFF
        self.id &= 0xfc0000ff
        self.id |= (_pgn << 8 & 0x3FFFF00)  # default pgn is None -> mypy reports error

    @property
    def j1939_tuple(self):  # type: () -> typing.Tuple[int, int, int]

        print("def : canmatrix - ArbitrationId - j1939_tuple")

        """Get tuple (destination, PGN, source)

        :rtype: tuple"""

        return self.j1939_destination, self.pgn, self.j1939_source

    @property
    def j1939_destination(self):

        print("def : canmatrix - ArbitrationId - j1939_destination")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        if self.j1939_pdu_format == 1:
            destination = self.j1939_ps
        else:
            destination = None
        return destination

    @property
    def j1939_source(self):

        print("def : canmatrix - ArbitrationId - j1939_source")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        return self.id & 0xFF

    @j1939_source.setter
    def j1939_source(self, value):  # type: (int) -> None

        print("def : canmatrix - ArbitrationId - j1939_source")

        self.extended = True
        self.id = (self.id & 0xffffff00) | (value & 0xff)

    @property
    def j1939_ps(self):

        print("def : canmatrix - ArbitrationId - j1939_ps")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        return (self.id >> 8) & 0xFF

    @property
    def j1939_pf(self):

        print("def : canmatrix - ArbitrationId - j1939_pf")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        return (self.id >> 16) & 0xFF

    @property
    def j1939_pdu_format(self):

        print("def : canmatrix - ArbitrationId - j1939_pdu_format")

        return 1 if (self.j1939_pf < 240) else 2

    @property
    def j1939_dp(self):

        print("def : canmatrix - ArbitrationId - j1939_dp")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        return (self.id >> 24) & 0x1

    @property
    def j1939_edp(self):

        print("def : canmatrix - ArbitrationId - j1939_edp")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        return (self.id >> 25) & 0x1

    @property
    def j1939_priority(self):

        print("def : canmatrix - ArbitrationId - j1939_priority")

        if not self.extended:
            raise J1939needsExtendedIdetifier
        return (self.id >> 26) & 0x7

    @j1939_priority.setter
    def j1939_priority(self, value):  # type: (int) -> None

        print("def : canmatrix - ArbitrationId - j1939_priority")

        self.extended = True
        self.id = (self.id & 0x3ffffff) | ((value & 0x7) << 26)

    @property
    def j1939_str(self):  # type: () -> str

        print("def : canmatrix - ArbitrationId - j1939_str")

        return "DA:0x{da:02X} PGN:0x{pgn:04X} SA:0x{sa:02X}".format(
            da=self.j1939_destination, pgn=self.pgn, sa=self.j1939_source)

    @classmethod
    def from_compound_integer(cls, i):  # type: (typing.Any) -> ArbitrationId

        # print("def : canmatrix - ArbitrationId - from_compound_integer - [{}]".format(i))

        return cls(
            id=i & cls.extended_id_mask,
            extended=(i & cls.compound_extended_mask) != 0,
        )

    @classmethod
    def from_pgn(cls, pgn):  # type: (int) -> ArbitrationId

        print("def : canmatrix - ArbitrationId - from_pgn")

        return cls(
            id=(pgn << 8), extended=True
        )

    def to_compound_integer(self):

        #print("def : canmatrix - ArbitrationId - to_compound_integer - {}".format(self.id))

        if self.extended:
            #print("def : canmatrix - ArbitrationId - to_compound_integer - 0x{:x}".format(self.id))
            return self.id | self.compound_extended_mask
        else:
            #print("def : canmatrix - ArbitrationId - to_compound_integer - 0x{:x}".format(self.id))
            return self.id

    def __eq__(self, other):

        print("def : canmatrix - ArbitrationId - __eq__")

        return (
                self.id == other.id
                and (
                        self.extended is None
                        or other.extended is None
                        or self.extended == other.extended
                )
        )


@attr.s(eq=False)
class Pdu(object):
    """
    Represents a PDU.

    PDUs are hierarchical groups of signals which are needed to represent Flexray busses
    Whereas a PDU is the same than a frame on CAN bus, at flexray a frame may consist of
    multiple PDUs (a bit like multiple signal layout for multiplexed can frames).
    This class is only used for flexray busses.
    """

    name = attr.ib(default="")  # type: str
    size = attr.ib(default=0)  # type: int
    id = attr.ib(default=0)  # type: int
    triggering_name = attr.ib(default="")  # type: str
    pdu_type = attr.ib(default="")  # type: str
    port_type = attr.ib(default="")  # type: str
    signals = attr.ib(factory=list)  # type: typing.MutableSequence[Signal]
    signalGroups = attr.ib(factory=list)  # type: typing.MutableSequence[SignalGroup]
    cycle_time = attr.ib(default=0)  # type: int

    def add_signal(self, signal):

        print("def : canmatrix - Pdu - add_signal")

        # type: (Signal) -> Signal
        """
        Add Signal to Pdu.

        :param Signal signal: Signal to be added.
        :return: the signal added.
        """
        self.signals.append(signal)
        return self.signals[len(self.signals) - 1]

    def add_signal_group(self, Name, Id, signalNames, e2e_trans=None):

        print("def : canmatrix - Pdu - add_signal_group")

        # type: (str, int, typing.Sequence[str]) -> None
        """Add new SignalGroup to the Frame. Add given signals to the group.

        :param str Name: Group name
        :param int Id: Group id
        :param list of str signalNames: list of Signal names to add. Non existing names are ignored.
        """
        newGroup = SignalGroup(Name, Id, e2e_trans=e2e_trans)
        self.signalGroups.append(newGroup)
        for signal in signalNames:
            signal = signal.strip()
            if signal.__len__() == 0:
                continue
            signalId = self.signal_by_name(signal)
            if signalId is not None:
                newGroup.add_signal(signalId)

    def get_signal_group_for_signal(self, signal_to_find):

        print("def : canmatrix - Pdu - get_signal_group_for_signal")

        for signal_group in self.signalGroups:
            for signal in signal_group:
                if signal == signal_to_find:
                    return signal_group
        return None

    def signal_by_name(self, name):

        print("def : canmatrix - Pdu - signal_by_name")

        # type: (str) -> typing.Union[Signal, None]
        """
        Get signal by name.

        :param str name: signal name to be found.
        :return: signal with given name or None if not found
        """
        for signal in self.signals:
            if signal.name == name:
                return signal
        return None


@attr.s(eq=False)
class Frame(object):
    """
    Represents CAN Frame.

    The Frame has  following mandatory attributes

    * arbitration_id,
    * name,
    * transmitters (list of ECU names),
    * size (DLC),
    * signals (list of signal-objects),
    * attributes (list of attributes),
    * receivers (list of ECU names),
    * comment

    and any *custom* attributes in `attributes` dict.

    Frame signals can be accessed using the iterator.
    """

    name = attr.ib(default="")  # type: str
    # mypy Unsupported converter:
    arbitration_id = attr.ib(converter=arbitration_id_converter, default=0)  # type: ArbitrationId
    size = attr.ib(default=0)  # type: int
    transmitters = attr.ib(factory=list)  # type: typing.MutableSequence[str]
    # extended = attr.ib(default=False)  # type: bool
    is_complex_multiplexed = attr.ib(default=False)  # type: bool
    is_fd = attr.ib(default=False)  # type: bool
    comment = attr.ib(default="")  # type: str
    # signals = attr.ib(factory=list)  # type: typing.MutableSequence[Signal]
    mux_names = attr.ib(factory=dict)  # type: typing.MutableMapping[int, str]
    attributes = attr.ib(factory=dict)  # type: typing.MutableMapping[str, typing.Any]
    receivers = attr.ib(factory=list)  # type: typing.MutableSequence[str]
    signalGroups = attr.ib(factory=list)  # type: typing.MutableSequence[SignalGroup]

    cycle_time = attr.ib(default=0)  # type: int

    is_j1939 = attr.ib(default=False)  # type: bool
    # ('cycleTime', '_cycleTime', int, None),
    # ('sendType', '_sendType', str, None),

    pdus = attr.ib(factory=list)  # type: typing.MutableSequence[Pdu]
    header_id = attr.ib(default=None)  # type: int

    # header_id

    @property
    def is_multiplexed(self):  # type: () -> bool

        print("def : canmatrix - Frame - is_multiplexed")

        """Frame is multiplexed if at least one of its signals is a multiplexer."""
        for sig in self.signals:
            if sig.is_multiplexer:
                return True
        return False

    @property
    def get_multiplexer(self):  # type: () -> typing.Union[Signal, None]

        print("def : canmatrix - Frame - get_multiplexer")

        """get multiplexer signal if any in frame."""
        for sig in self.signals:
            if sig.is_multiplexer:
                return sig
        return None

    @property
    def get_multiplexer_values(self):  # type: () -> typing.Sequence[int]

        print("def : canmatrix - Frame - get_multiplexer_values")

        """get possible multiplexer values."""
        multiplexer_values = {
            sig.mux_val
            for sig in self.signals
            if sig.mux_val is not None
        }
        return list(multiplexer_values)

    def get_signals_for_multiplexer_value(self, mux_value):

        print("def : canmatrix - Frame - get_signals_for_multiplexer_value")

        # type: (int) -> typing.Sequence[Signal]
        """Find Frame Signals by given muxer value.
        :param int mux_value: muxer value
        :return: list of signals relevant for given muxer value.
        :rtype: list of signals
        """
        muxed_signals = []
        for sig in self.signals:
            if (sig.mux_val is None and not sig.is_multiplexer) or sig.mux_val == mux_value:
                muxed_signals.append(sig)
        return muxed_signals

    @property
    def is_pdu_container(self):  # type: () -> bool

        print("def : canmatrix - Frame - is_pdu_container")

        return len(self.pdus) > 0

    @property
    def get_pdu_id_values(self):  # type: () -> typing.Sequence[int]

        print("def : canmatrix - Frame - get_pdu_id_values")

        return list({pdu.id for pdu in self.pdus})

    @property
    def pgn(self):  # type: () -> int

        print("def : canmatrix - Frame - pgn")

        return self.arbitration_id.pgn

    @pgn.setter
    def pgn(self, value):  # type: (int) -> None

        print("def : canmatrix - Frame - pgn")

        self.arbitration_id.pgn = value

    @property
    def priority(self):  # type: () -> int

        print("def : canmatrix - Frame - priority")

        """Get J1939 priority."""
        return self.arbitration_id.j1939_priority

    @priority.setter
    def priority(self, value):  # type: (int) -> None

        print("def : canmatrix - Frame - priority")

        """Set J1939 priority."""
        self.arbitration_id.j1939_priority = value

    @property
    def source(self):  # type: () -> int

        print("def : canmatrix - Frame - source")

        """Get J1939 source."""
        return self.arbitration_id.j1939_source

    @source.setter
    def source(self, value):  # type: (int) -> None

        print("def : canmatrix - Frame - source")

        """Set J1939 source."""
        self.arbitration_id.j1939_source = value

    @property
    def effective_cycle_time(self):

        print("def : canmatrix - Frame - effective_cycle_time")

        """Calculate effective cycle time for frame, depending on singal cycle times"""
        min_cycle_time_list = [y for y in [x.cycle_time for x in self.signals] + [self.cycle_time] if y != 0]
        if len(min_cycle_time_list) == 0:
            return 0
        elif len(min_cycle_time_list) == 1:
            return min_cycle_time_list[0]
        else:
            gcd = canmatrix.utils.get_gcd(min_cycle_time_list[0], min_cycle_time_list[1])
            for i in range(2, len(min_cycle_time_list)):
                gcd = canmatrix.utils.get_gcd(gcd, min_cycle_time_list[i])
            return gcd
        #    return min(min_cycle_time_list)

    # @property
    # def sendType(self, db = None):
    #    if self._sendType is None:
    #        self._sendType = self.attribute("GenMsgSendType")
    #    return self._sendType
    #
    # @sendType.setter
    # def sendType(self, value):
    #    self._sendType = value

    def attribute(self, attribute_name, db=None, default=None):

        print("def : canmatrix - Frame - attribute")

        # type: (str, typing.Optional[CanMatrix], typing.Any) -> typing.Any
        """Get any Frame attribute by its name.

        :param str attribute_name: attribute name, can be mandatory (ex: id) or optional (customer) attribute.
        :param CanMatrix db: Optional database parameter to get global default attribute value.
        :param default: Default value if attribute doesn't exist.
        :return: Return the attribute value if found, else `default` or None
        """
        if attribute_name in attr.fields_dict(type(self)):
            return getattr(self, attribute_name)
        if attribute_name in self.attributes:
            return self.attributes[attribute_name]
        elif db is not None and attribute_name in db.frame_defines:
            define = db.frame_defines[attribute_name]
            return define.defaultValue
        return default

    def add_signal_group(self, Name, Id, signalNames, e2e_trans=None):

        print("def : canmatrix - Frame - add_signal_group")

        # type: (str, int, typing.Sequence[str]) -> None
        """Add new SignalGroup to the Frame. Add given signals to the group.

        :param str Name: Group name
        :param int Id: Group id
        :param list of str signalNames: list of Signal names to add. Non existing names are ignored.
        """
        newGroup = SignalGroup(Name, Id, e2e_trans=e2e_trans)
        #print("def : canmatrix - Frame - add_signal_group - 1")
        self.signalGroups.append(newGroup)
        #print("def : canmatrix - Frame - add_signal_group - 2")

        for signal in signalNames:
            #print("def : canmatrix - Frame - add_signal_group - 3")
            # signal = signal.strip()
            # if signal.__len__() == 0:
            #     continue
            #
            # signalId = self.signal_by_name(signal)

            print("def : canmatrix - Frame - add_signal_group - FILL : {}".format(signal))
            newGroup.add_signal(signal)
            # if signalId is not None:
            #     newGroup.add_signal(signalId)
            #     print("def : canmatrix - Frame - add_signal_group - end")

        # for signal in signalNames:
        #     #print("def : canmatrix - Frame - add_signal_group - 3")
        #     signal = signal.strip()
        #     if signal.__len__() == 0:
        #         continue
        #
        #     signalId = self.signal_by_name(signal)
        #
        #     print("def : canmatrix - Frame - add_signal_group - FILL : {}".format(signal))
        #
        #     if signalId is not None:
        #         newGroup.add_signal(signalId)
        #         print("def : canmatrix - Frame - add_signal_group - end")

    def signal_group_by_name(self, name):

        print("def : canmatrix - Frame - signal_group_by_name")

        # type: (str) -> typing.Union[SignalGroup, None]
        """Get signal group.

        :param str name: group name
        :return: SignalGroup by name or None if not found.
        :rtype: SignalGroup
        """
        for signalGroup in self.signalGroups:
            if signalGroup.name == name:
                return signalGroup
        return None

    def add_pdu(self, pdu):

        print("def : canmatrix - Frame - add_pdu")

        # type: (Pdu) -> Pdu
        """
        Add Pdu to Frame.

        :param Pdu pdu: Pdu to be added.
        :return: the pdu added.
        """
        self.pdus.append(pdu)
        return self.pdus[len(self.pdus) - 1]

    def pdu_by_name(self, name):

        print("def : canmatrix - Frame - pdu_by_name")

        # type: (str) -> typing.Union[Pdu, None]
        """Get PDU.

        :param str name: PDU name
        :return: PDU by name or None if not found.
        :rtype: Pdu
        """
        for pdu in self.pdus:
            if pdu.name == name:
                return pdu
        return None

    def pdu_by_id(self, pdu_id):

        print("def : canmatrix - Frame - pdu_by_id")

        # type: (int) -> typing.Union[Pdu, None]
        """Get PDU.

        :param int pdu_id: PDU id
        :return: PDU by id or None if not found.
        :rtype: Pdu
        """
        for pdu in self.pdus:
            if pdu.id == pdu_id:
                return pdu
        return None

    def add_signal(self, signal):

        print("def : canmatrix - Frame - add_signal - [{}]".format(signal))

        # type: (Signal) -> Signal
        """
        Add Signal to Frame.

        :param Signal signal: Signal to be added.
        :return: the signal added.
        """
        self.signals.append(signal)
        return self.signals[len(self.signals) - 1]

    def add_transmitter(self, transmitter):

        print("def : canmatrix - Frame - add_transmitter - [{}]".format(transmitter))

        # type: (str) -> None
        """Add transmitter ECU Name to Frame.

        :param str transmitter: transmitter name
        """
        if transmitter not in self.transmitters:
            self.transmitters.append(transmitter)

    def del_transmitter(self, transmitter):

        print("def : canmatrix - Frame - del_transmitter")

        # type: (str) -> None
        """Delete transmitter ECU Name from Frame.

        :param str transmitter: transmitter name
        """
        if transmitter in self.transmitters:
            self.transmitters.remove(transmitter)

    def add_receiver(self, receiver):

        print("def : canmatrix - Frame - add_receiver - [{}]".format(receiver))

        # type: (str) -> None
        """Add receiver ECU Name to Frame.

        :param str receiver: receiver name
        """
        if receiver not in self.receivers:
            self.receivers.append(receiver)

    def signal_by_name(self, name):

        print("def : canmatrix - Frame - signal_by_name - [{}]".format(name))
        """
        Get signal by name.

        :param str name: signal name to be found.
        :return: signal with given name or None if not found
        """
        for signal in self.signals:
            #print("def : canmatrix - Frame - signal_by_name SEARCH - ")
            if signal.name == name:
                return signal
        return None

    def glob_signals(self, glob_str):

        print("def : canmatrix - Frame - glob_signals")

        # type: (str) -> typing.Sequence[Signal]
        """Find Frame Signals by given glob pattern.

        :param str glob_str: glob pattern for signal name. See `fnmatch.fnmatchcase`
        :return: list of Signals by glob pattern.
        :rtype: list of Signal
        """
        return_array = []
        for signal in self.signals:
            if fnmatch.fnmatchcase(signal.name, glob_str):
                return_array.append(signal)
        return return_array

    def add_attribute(self, attribute, value):

        print("def : canmatrix - Frame - add_attribute - [{},{},{}]".format(self.name,attribute,value))

        # type: (str, typing.Any) -> None
        """
        Add the attribute with value to customer Frame attribute-list. If Attribute already exits, modify its value.
        :param str attribute: Attribute name
        :param any value: attribute value
        """
        try:
            self.attributes[attribute] = str(value)
        except UnicodeDecodeError:
            self.attributes[attribute] = value
        if type(self.attributes[attribute]) == str:
            self.attributes[attribute] = self.attributes[attribute].strip()

    def del_attribute(self, attribute):

        print("def : canmatrix - Frame - del_attribute")

        # type: (str) -> typing.Any
        """
        Remove attribute from customer Frame attribute-list.

        :param str attribute: Attribute name
        """
        if attribute in self.attributes:
            del self.attributes[attribute]

    def add_comment(self, comment):

        print("def : canmatrix - Frame - add_comment")

        # type: (str) -> None
        """
        Set Frame comment.

        :param str comment: Frame comment
        """
        self.comment = comment

    def calc_dlc(self):

        print("def : canmatrix - Frame - calc_dlc")

        # type: () -> None
        """
        Compute minimal Frame DLC (length) based on its Signals

        :return: Message DLC
        """
        max_bit = 0
        for sig in self.signals:
            if sig.get_startbit() + int(sig.size) > max_bit:
                max_bit = sig.get_startbit() + int(sig.size)
        max_byte = (max_bit + 7) // 8
        if self.is_pdu_container:
            max_byte *= len(self.pdus)
            for pdu in self.pdus:
                max_byte += pdu.size
        self.size = max(self.size, max_byte)

    def fit_dlc(self):

        print("def : canmatrix - Frame - fit_dlc")

        """
            Compute next allowed DLC (length) for current Frame
        """
        max_byte = self.size
        last_size = 8
        for max_size in [12, 16, 20, 24, 32, 48, 64]:
            if max_byte > last_size and max_byte < max_size:
                self.size = max_size
                break
            last_size = max_size

    def get_frame_layout(self):

        print("def : canmatrix - Frame - get_frame_layout")

        # type: () -> typing.Sequence[typing.Sequence[str]]
        """
        get layout of frame.

        Represents the bit usage in the frame by means of a list with n items (n bits of frame length).
        Every item represents one bit and contains a list of signals (object refs) with each signal, occupying that bit.
        Bits with empty list are unused.

        Example: [[], [], [], [sig1], [sig1], [sig1, sig5], [sig2, sig5], [sig2], []]
        :return: list of lists with signalnames
        :rtype: list of lists
        """
        little_bits = [[] for _dummy in range((self.size * 8))]  # type: typing.List[typing.List]
        big_bits = [[] for _dummy in range((self.size * 8))]  # type: typing.List[typing.List]
        for signal in self.signals:
            if signal.is_little_endian:
                least = len(little_bits) - signal.start_bit
                most = least - signal.size
                for little_bit_signals in little_bits[most:least]:
                    little_bit_signals.append(signal)

            else:
                most = signal.start_bit
                least = most + signal.size
                for big_bit_signals in big_bits[most:least]:
                    big_bit_signals.append(signal)

        little_bits_iter = reversed(tuple(grouper(little_bits, 8)))
        little_bits = list(itertools.chain(*little_bits_iter))

        return_list = [
            little + big
            for little, big in zip(little_bits, big_bits)
        ]

        return return_list

    def set_fd_type(self):  # type: () -> None

        #print("def : canmatrix - Frame - set_fd_type")

        """Try to guess and set the CAN type for every frame.

        If a Frame is longer than 8 bytes, it must be Flexible Data Rate frame (CAN-FD).
        If not, the Frame type stays unchanged.
        """
        if self.size > 8:
            self.is_fd = True

    def create_dummy_signals(self):  # type: () -> None

        print("def : canmatrix - Frame - create_dummy_signals")

        """Create big-endian dummy signals for unused bits.

        Names of dummy signals are *_Dummy_<frame.name>_<index>*
        """
        bitfield = self.get_frame_layout()
        startBit = -1
        sigCount = 0
        for index, bit_signals in enumerate(bitfield):
            if bit_signals == [] and startBit == -1:
                startBit = index
            if (index == (len(bitfield) - 1) or bit_signals != []) and startBit != -1:
                if index == (len(bitfield) - 1):
                    index = len(bitfield)
                self.add_signal(
                    Signal("_Dummy_%s_%d" % (self.name, sigCount), size=index - startBit, start_bit=startBit,
                           is_little_endian=False))
                startBit = -1
                sigCount += 1

    def signals_to_bytes(self, data):

        print("def : canmatrix - Frame - signals_to_bytes")

        # type: (typing.Mapping[str, canmatrix.types.RawValue]) -> bytes
        """Return a byte string containing the values from data packed
        according to the frame format.

        :param data: data dictionary of signal : rawValue
        :return: A byte string of the packed values.
        """

        little_bits = [None] * (self.size * 8)  # type: typing.List[typing.Optional[str]]
        big_bits = list(little_bits)
        for signal in self.signals:
            if signal.name in data:
                value = data.get(signal.name)
                bits = pack_bitstring(signal.size, signal.is_float, value, signal.is_signed)

                if signal.is_little_endian:
                    least = self.size * 8 - signal.start_bit
                    most = least - signal.size

                    little_bits[most:least] = bits
                else:
                    most = signal.start_bit
                    least = most + signal.size

                    big_bits[most:least] = bits
        little_bits_iter = reversed(tuple(grouper(little_bits, 8)))
        little_bits = list(itertools.chain(*little_bits_iter))
        bitstring = ''.join(
            next(x for x in (l, b, '0') if x is not None)
            # l if l != ' ' else (b if b != ' ' else '0')
            for l, b in zip(little_bits, big_bits)
        )
        return bytearray(
            int(''.join(b), 2)
            for b in grouper(bitstring, 8)
        )

    def encode(self, data=None):

        print("def : canmatrix - Frame - encode")

        # type: (typing.Optional[typing.Mapping[str, typing.Any]]) -> bytes
        """Return a byte string containing the values from data packed
        according to the frame format.

        :param dict data: data dictionary
        :return: A byte string of the packed values.
        """

        data = dict() if data is None else data

        if self.is_complex_multiplexed:
            raise EncodingComplexMultiplexed
        elif self.is_pdu_container:
            raise EncodingConatainerPdu  # TODO add encoding
        elif self.is_multiplexed:
            # search for mulitplexer-signal
            for signal in self.signals:
                if signal.is_multiplexer:
                    muxSignal = signal
                    muxVal = data.get(signal.name)
                    break
            else:
                raise MissingMuxSignal
            # create list of signals which belong to muxgroup
            encodeSignals = [muxSignal.name]
            for signal in self.signals:
                if signal.mux_val == muxVal or signal.mux_val is None:
                    encodeSignals.append(signal.name)
            newData = dict()
            # kick out signals, which do not belong to this mux-id
            for signalName in data:
                if signalName in encodeSignals:
                    newData[signalName] = data[signalName]
            data = newData
        return self.signals_to_bytes(data)

    @staticmethod
    def bytes_to_bitstrings(data):

        print("def : canmatrix - Frame - bytes_to_bitstrings")

        # type: (bytes) -> typing.Tuple[str, str]
        """Return two arrays big and little containing bits of given data (bytearray)

        :param data: bytearray of bits (little endian).
            i.e. bytearray([0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8])
        :return: bit arrays in big and little byteorder
        """
        b = tuple('{:08b}'.format(b) for b in data)
        little = ''.join(reversed(b))
        big = ''.join(b)

        return little, big

    @staticmethod
    def bitstring_to_signal_list(signals, big, little, size):

        print("def : canmatrix - Frame - bitstring_to_signal_list")

        # type: (typing.Sequence[Signal], str, str, int) -> typing.Sequence[canmatrix.types.RawValue]
        """Return OrderedDictionary with Signal Name: object decodedSignal (flat / without support for multiplexed frames)

        :param signals: Iterable of signals (class signal) to decode from frame.
        :param big: bytearray of bits (big endian).
        :param little: bytearray of bits (little endian).
        :param size: number of bits.
        :return: array with raw values (same order like signals)
        """
        unpacked = []
        for signal in signals:
            if signal.is_little_endian:
                least = size - signal.start_bit
                most = least - signal.size

                bits = little[most:least]
            else:
                most = signal.start_bit
                least = most + signal.size

                bits = big[most:least]
            unpacked.append(unpack_bitstring(signal.size, signal.is_float, signal.is_signed, bits))

        return unpacked

    def unpack(self, data, report_error=True):

        print("def : canmatrix - Frame - unpack")

        # type: (bytes, bool) -> typing.Mapping[str, DecodedSignal]
        """Return OrderedDictionary with Signal Name: object decodedSignal (flat / without support for multiplexed frames)
        decodes every signal in signal-list.

        :param data: bytearray
            i.e. bytearray([0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8])
        :param report_error: set to False to silence error output
        :return: OrderedDictionary
        """

        rx_length = len(data)
        if rx_length != self.size and report_error:
            raise DecodingFrameLength(
                'Received message 0x{self.arbitration_id.id:08X} '
                'with length {rx_length}, expected {self.size}'.format(**locals())
            )
        elif self.is_pdu_container:
            header_id_signal = self.signal_by_name("Header_ID")
            header_dlc_signal = self.signal_by_name("Header_DLC")
            if header_id_signal is None or header_dlc_signal is None:
                raise DecodingConatainerPdu(
                    'Received message 0x{:08X} without Header_ID or '
                    'Header_DLC signal'.format(self.arbitration_id.id)
                )
            # TODO: may be we need to check that ID/DLC signals are contiguous
            header_size = header_id_signal.size + header_dlc_signal.size
            little, big = self.bytes_to_bitstrings(data)
            size = self.size * 8
            return_dict = dict({"pdus": []})
            # decode signal which are not in PDUs
            signals = [s for s in self.signals if s not in [header_id_signal, header_dlc_signal]]
            if signals:
                unpacked = self.bitstring_to_signal_list(signals, big, little, size)
                for s, v in zip(signals, unpacked):
                    return_dict[s.name] = DecodedSignal(v, s)
            # decode PDUs
            offset = header_id_signal.start_bit
            header_signals = [header_id_signal, header_dlc_signal]
            while (offset + header_size) < size:
                unpacked = self.bitstring_to_signal_list(
                    header_signals,
                    big[offset:offset + header_size],
                    little[size - offset - header_size:size - offset],
                    header_size
                )
                offset += header_size
                pdu_id = unpacked[0]
                pdu_dlc = unpacked[1]
                for s, v in zip(header_signals, unpacked):
                    if s.name not in return_dict:
                        return_dict[s.name] = []
                    return_dict[s.name].append(DecodedSignal(v, s))
                pdu = self.pdu_by_id(pdu_id)
                if pdu is None:
                    return_dict['pdus'].append(None)
                else:
                    unpacked = self.bitstring_to_signal_list(
                        pdu.signals,
                        big[offset:offset + pdu_dlc * 8],
                        little[size - offset - pdu_dlc * 8:size - offset],
                        pdu_dlc * 8
                    )
                    pdu_dict = dict()
                    for s, v in zip(pdu.signals, unpacked):
                        pdu_dict[s.name] = DecodedSignal(v, s)
                    return_dict["pdus"].append({pdu.name: pdu_dict})
                offset += (pdu_dlc * 8)
            return return_dict
        else:
            little, big = self.bytes_to_bitstrings(data)

            unpacked = self.bitstring_to_signal_list(self.signals, big, little, self.size * 8)

            return_dict = dict()

            for s, v in zip(self.signals, unpacked):
                return_dict[s.name] = DecodedSignal(v, s)

            return return_dict

    def _get_sub_multiplexer(self, parent_multiplexer_name, parent_multiplexer_value):

        print("def : canmatrix - Frame - _get_sub_multiplexer")

        """
        get any sub-multiplexer in frame used
        for complex-multiplexed frame decoding

        :param parent_multiplexer_name: string with name of parent multiplexer
        :param parent_multiplexer_value: raw_value (int) of parent multiplexer
        :return: muxer signal or None
        """
        for signal in self.signals:
            if signal.is_multiplexer and signal.muxer_for_signal == parent_multiplexer_name and signal.multiplexer_value_in_range(
                    parent_multiplexer_value):
                return signal

    def _filter_signals_for_multiplexer(self, multiplexer_name, multiplexer_value):

        print("def : canmatrix - Frame - _filter_signals_for_multiplexer")

        """
        filter a list of signals with given multiplexer (name and value)
        used for complex-multiplexed frame decoding


        :param multiplexer_name: string with name of parent multiplexer
        :param multiplexer_value: raw_value (int) of parent multiplexer
        :return: filtered array of canmatrix.Signal
        """
        filtered_signals = []
        for signal in self.signals:
            if signal.multiplexer_value_in_range(
                    multiplexer_value) and signal.muxer_for_signal == multiplexer_name and not signal.is_multiplexer:
                filtered_signals.append(signal)
            elif signal.name == multiplexer_name:
                filtered_signals.append(signal)
        return filtered_signals

    def decode(self, data):

        print("def : canmatrix - Frame - decode")

        # type: (bytes) -> typing.Mapping[str, typing.Any]
        """Return OrderedDictionary with Signal Name: object decodedSignal (support for multiplexed frames)
        decodes only signals matching to muxgroup

        :param data: bytearray .
            i.e. bytearray([0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8])
        :return: OrderedDictionary
        """
        decoded = self.unpack(data)

        if self.is_complex_multiplexed:
            decoded_values = dict()
            filtered_signals = self._filter_signals_for_multiplexer(None, None)

            multiplex_name = None
            multiplex_value = None

            sub_multiplexer = self._get_sub_multiplexer(multiplex_name, multiplex_value)
            while sub_multiplexer is not None:
                multiplex_name = sub_multiplexer.name
                multiplex_signal = decoded_values[multiplex_name] = decoded[multiplex_name]
                multiplex_value = multiplex_signal.raw_value
                filtered_signals += self._filter_signals_for_multiplexer(multiplex_name, multiplex_value)

                sub_multiplexer = self._get_sub_multiplexer(multiplex_name, multiplex_value)

            for signal in filtered_signals:
                decoded_values[signal.name] = decoded[signal.name]
            return decoded_values

        elif self.is_multiplexed and not self.is_pdu_container:
            decoded_values = dict()
            # find multiplexer and decode only its value:

            for signal in self.signals:
                if signal.is_multiplexer:
                    muxVal = decoded[signal.name].raw_value

            # find all signals with the identified multiplexer-value
            for signal in self.signals:
                if signal.mux_val == muxVal or signal.mux_val is None:
                    decoded_values[signal.name] = decoded[signal.name]
            return decoded_values

        else:
            return decoded

    def __str__(self):  # type: () -> str

        # print("def : canmatrix - Frame - __str__")

        """Represent the frame by its name only."""
        return self.name  # add more details than the name only?


class Define(object):
    """
    Hold the defines and default-values.
    """

    def __init__(self, definition):  # type (str) -> None

        # print("def : canmatrix - Define - __init__")

        """Initialize Define object.

        :param str definition: definition string. Ex: "INT -5 10"
        """
        definition = definition.strip()
        self.definition = definition
        self.type = None  # type: typing.Optional[str]
        self.defaultValue = None  # type: typing.Any

        def safe_convert_str_to_int(inStr):  # type: (str) -> int

            # print("def : canmatrix - Define - safe_convert_str_to_int")

            """Convert string to int safely. Check that it isn't float.

            :param str inStr: integer represented as string.
            :rtype: int
            """
            out = int(defaultFloatFactory(inStr))
            if out != defaultFloatFactory(inStr):
                logger.warning(
                    "Warning, integer was expected but got float: got: {0} using {1}\n".format(inStr, str(out)))
            return out

        # for any known type:
        if definition[0:3] == 'INT':
            self.type = 'INT'
            min, max = definition[4:].split(' ', 2)
            self.min = safe_convert_str_to_int(min)
            self.max = safe_convert_str_to_int(max)
            self.max = safe_convert_str_to_int(max)

        elif definition[0:6] == 'STRING':
            self.type = 'STRING'
            self.min = None
            self.max = None

        elif definition[0:4] == 'ENUM':
            self.type = 'ENUM'
            tempValues = canmatrix.utils.quote_aware_comma_split(definition[5:])
            self.values = []  # type: typing.List[str]
            for value in tempValues:
                value = value.replace("vector_leerstring", "")
                self.values.append(value)

        elif definition[0:3] == 'HEX':  # differently rendered in DBC editor, but values are saved like for an INT
            self.type = 'HEX'
            min, max = definition[4:].split(' ', 2)
            self.min = safe_convert_str_to_int(min)
            self.max = safe_convert_str_to_int(max)

        elif definition[0:5] == 'FLOAT':
            self.type = 'FLOAT'
            min, max = definition[6:].split(' ', 2)
            self.min = defaultFloatFactory(min)
            self.max = defaultFloatFactory(max)

    def set_default(self, default):  # type: (typing.Any) -> None

        print("def : canmatrix - Define - set_default")

        """Set Definition default value.

        :param default: default value; number, str or quoted str ("value")
        """
        if default is not None and len(default) > 1 and default[0] == '"' and default[-1] == '"':
            default = str(default[1:-1])
        elif default.isdigit():
            default = int(default)
        self.defaultValue = default

    def update(self):  # type: () -> None

        print("def : canmatrix - Define - update")

        """Update definition string for type ENUM.

        For type ENUM rebuild the definition string from current values. Otherwise do nothing.
        """
        if self.type != 'ENUM':
            return
        self.definition = 'ENUM "' + '","'.join(self.values) + '"'


import enum


class matrix_class(enum.Enum):
    CAN = 1
    FLEXRAY = 2
    SOMEIP = 3


@attr.s(eq=False)
class CanMatrix(object):
    """
    The Can-Matrix-Object
    attributes (global canmatrix-attributes),
    ecus (list of ECUs),
    frames (list of Frames)
    signal_defines (list of signal-attribute types)
    frame_defines (list of frame-attribute types)
    ecu_defines (list of ECU-attribute types)
    global_defines (list of global attribute types)
    value_tables (global defined values)
    """

    type = attr.ib(default=matrix_class.CAN)  # type: matrix_class
    attributes = attr.ib(factory=dict)  # type: typing.MutableMapping[str, typing.Any]
    ecus = attr.ib(factory=list)  # type: typing.MutableSequence[Ecu]
    frames = attr.ib(factory=list)  # type: typing.MutableSequence[Frame]
    signals = attr.ib(factory=list)  # type: typing.MutableSequence[Signal]

    signal_defines = attr.ib(factory=dict)  # type: typing.MutableMapping[str, Define]
    frame_defines = attr.ib(factory=dict)  # type: typing.MutableMapping[str, Define]
    global_defines = attr.ib(factory=dict)  # type: typing.MutableMapping[str, Define]
    env_defines = attr.ib(factory=dict)  # type: typing.MutableMapping[str, Define]
    ecu_defines = attr.ib(factory=dict)  # type: typing.MutableMapping[str, Define]
    value_tables = attr.ib(factory=dict)  # type: typing.MutableMapping[str, typing.MutableMapping]
    env_vars = attr.ib(factory=dict)  # type: typing.MutableMapping[str, typing.MutableMapping]

    baudrate = attr.ib(default=0)  # type:int
    fd_baudrate = attr.ib(default=0)  # type:int

    load_errors = attr.ib(factory=list)  # type: typing.MutableSequence[Exception]

    def __iter__(self):  # type: () -> typing.Iterator[Frame]

        print("def : canmatrix - CanMatrix - __iter__")

        """Matrix iterates over Frames (Messages)."""
        return iter(self.frames)

    def add_env_var(self, name, envVarDict):  # type: (str, typing.MutableMapping) -> None

        print("def : canmatrix - CanMatrix - add_env_var")

        self.env_vars[name] = envVarDict

    def add_env_attribute(self, env_name, attribute_name, attribute_value):

        print("def : canmatrix - CanMatrix - add_env_attribute")

        # type: (str, str, typing.Any) -> None
        if env_name in self.env_vars:
            if not "attributes" in self.env_vars[env_name]:
                self.env_vars[env_name]["attributes"] = dict()
            self.env_vars[env_name]["attributes"][attribute_name] = attribute_value

    @property
    def contains_fd(self):  # type: () -> bool

        print("def : canmatrix - CanMatrix - contains_fd")

        for signal in self.signals:
            frame = signal.frames
            if frame.is_fd:
                return True
        return False

    @property
    def contains_j1939(self):  # type: () -> bool

        print("def : canmatrix - CanMatrix - contains_j1939")

        """Check whether the Matrix contains any J1939 Frame."""
        for frame in self.frames:
            if frame.is_j1939:
                return True
        return False

    def attribute(self, attributeName, default=None):  # type(str, typing.Any) -> typing.Any

        print("def : canmatrix - CanMatrix - attribute")

        """Return custom Matrix attribute by name.

        :param str attributeName: attribute name
        :param default: default value if given attribute doesn't exist
        :return: attribute value or default or None if no such attribute found.
        """
        if attributeName in self.attributes:
            return self.attributes[attributeName]
        elif attributeName in self.global_defines:
            define = self.global_defines[attributeName]
            return define.defaultValue
        else:
            return default

    def add_value_table(self, name, valueTable):  # type: (str, typing.Mapping) -> None

        print("def : canmatrix - CanMatrix - add_value_table [{},{}]".format(name,valueTable))

        """Add named value table.

        :param str name: value table name
        :param valueTable: value table itself
        """
        self.value_tables[name] = normalize_value_table(valueTable)

    def add_attribute(self, attribute, value):  # type: (str, typing.Any) -> None

        print("def : canmatrix - CanMatrix - add_attribute - [{},{}]".format(attribute,value))

        """
        Add attribute to Matrix attribute-list.

        :param str attribute: attribute name
        :param value: attribute value
        """
        self.attributes[attribute] = value

    def add_signal_defines(self, type, definition):

        print("def : canmatrix - CanMatrix - add_signal_defines - [{},{}]".format(type,definition))

        """
        Add signal-attribute definition to canmatrix.

        :param str type: signal type
        :param str definition: signal-attribute string definition, see class Define
        """
        if type not in self.signal_defines:
            # print(Define(definition))
            self.signal_defines[type] = Define(definition)

    def add_frame_defines(self, name, definition):  # type: (str, str) -> None

        print("def : canmatrix - CanMatrix - add_frame_defines - [{},{},{}]".format(self.frames,name,definition))

        """
        Add frame-attribute definition to canmatrix.

        :param str name: frame type
        :param str definition: frame definition as string
        """
        if name not in self.frame_defines:
            self.frame_defines[name] = Define(definition)

    def add_ecu_defines(self, name, definition):  # type: (str, str) -> None

        print("def : canmatrix - CanMatrix - add_ecu_defines - [{},{}]".format(name,definition))

        """
        Add Boardunit-attribute definition to canmatrix.

        :param str name: Boardunit type
        :param str definition: BU definition as string
        """
        if name not in self.ecu_defines:
            #print("def : canmatrix - CanMatrix - add_ecu_defines - {} - {}".format(name, definition))
            self.ecu_defines[name] = Define(definition)
            #print("def : canmatrix - CanMatrix - add_ecu_defines - CHECK - {}".format(self.ecu_defines[name].min))

    def add_env_defines(self, name, definition):  # type: (str, str) -> None

        print("def : canmatrix - CanMatrix - add_env_defines - {} - {}".format(name,definition))

        """
        Add enviroment variable-attribute definition to canmatrix.

        :param str name: enviroment variable type
        :param str definition: enviroment variable definition as string
        """
        if name not in self.env_defines:
            self.env_defines[name] = Define(definition)

    def add_global_defines(self, name, definition):  # type: (str, str) -> None

        print("def : canmatrix - CanMatrix - add_global_defines - [{},{}]".format(name,definition))

        """
        Add global-attribute definition to canmatrix.

        :param str name: attribute type
        :param str definition: attribute definition as string
        """
        if name not in self.global_defines:
            self.global_defines[name] = Define(definition)

    def add_define_default(self, name, value):  # type: (str, typing.Any) -> None

        print("def : canmatrix - CanMatrix - add_define_default - ",type(value))

        if name in self.signal_defines:
            self.signal_defines[name].set_default(value)
        if name in self.frame_defines:
            self.frame_defines[name].set_default(value)
        if name in self.ecu_defines:
            self.ecu_defines[name].set_default(value)
        if name in self.global_defines:
            self.global_defines[name].set_default(value)

    def delete_obsolete_ecus(self):  # type: () -> None

        print("def : canmatrix - CanMatrix - delete_obsolete_ecus")

        """Delete all unused ECUs
        """
        used_ecus = [ecu for f in self.frames for ecu in f.transmitters]
        used_ecus += [ecu for f in self.frames for ecu in f.receivers]
        used_ecus += [ecu for f in self.frames for s in f.signals for ecu in s.receivers]
        used_ecus += [ecu for s in self.signals for ecu in s.receivers]
        ecus_to_delete = [ecu.name for ecu in self.ecus if ecu.name not in used_ecus]
        for ecu in ecus_to_delete:
            self.del_ecu(ecu)

    def delete_obsolete_defines(self):  # type: () -> None

        print("def : canmatrix - CanMatrix - delete_obsolete_defines")

        """Delete all unused Defines.

        Delete them from frame_defines, ecu_defines and signal_defines.
        """
        defines_to_delete = set()  # type: typing.Set[str]
        for frameDef in self.frame_defines:
            for frame in self.frames:
                if frameDef in frame.attributes:
                    break
            else:
                defines_to_delete.add(frameDef)
        for element in defines_to_delete:
            del self.frame_defines[element]
        defines_to_delete = set()
        for ecu_define in self.ecu_defines:
            for ecu in self.ecus:
                if ecu_define in ecu.attributes:
                    break
            else:
                defines_to_delete.add(ecu_define)
        for element in defines_to_delete:
            del self.ecu_defines[element]

        defines_to_delete = set()
        for signal_define in self.signal_defines:
            for frame in self.frames:
                for signal in frame.signals:
                    if signal_define in signal.attributes:
                        break
                else:
                    defines_to_delete.add(signal_define)
        for element in defines_to_delete:
            del self.signal_defines[element]

    def frame_by_id(self, arbitration_id):  # type: (ArbitrationId) -> typing.Union[Frame, None]

        print("def : canmatrix - CanMatrix - frame_by_id")

        """Get Frame by its arbitration id.

        :param ArbitrationId arbitration_id: Frame id as canmatrix.ArbitrationId
        :rtype: Frame or None
        """
        for test in self.frames:
            if test.arbitration_id == arbitration_id:
                # found ID while ignoring extended or standard
                return test
        return None

    def frame_by_pgn(self, pgn):  # type: (int) -> typing.Union[Frame, None]

        print("def : canmatrix - CanMatrix - frame_by_pgn")

        """Get Frame by pgn (j1939).n

        :param int pgn: pgn to search for
        :rtype: Frame or None
        """

        for test in self.frames:
            if test.arbitration_id.pgn == canmatrix.ArbitrationId.from_pgn(pgn).pgn:
                # canmatrix.ArbitrationId.from_pgn(pgn).pgn instead
                # of just pgn is needed to do the pf >= 240 check
                return test
        return None

    def frame_by_name(self, name):  # type: (str) -> typing.Union[Frame, None]

        print("def : canmatrix - CanMatrix - frame_by_name")

        """Get Frame by name.

        :param str name: Frame name to search for
        :rtype: Frame or None
        """
        for test in self.frames:
            if test.name == name:
                return test
        return None

    def glob_frames(self, globStr):  # type: (str) -> typing.List[Frame]

        print("def : canmatrix - CanMatrix - glob_frames")

        """Find Frames by given glob pattern.

        :param str globStr: glob pattern to filter Frames. See `fnmatch.fnmatchcase`.
        :rtype: list of Frame
        """
        return_array = []
        for test in self.frames:
            if fnmatch.fnmatchcase(test.name, globStr):
                return_array.append(test)
        return return_array

    def ecu_by_name(self, name):  # type: (str) -> typing.Union[Ecu, None]

        print("def : canmatrix - CanMatrix - ecu_by_name")

        """
        Returns Boardunit by Name.

        :param str name: BoardUnit name
        :rtype: Ecu or None
        """
        for test in self.ecus:
            if test.name == name:
                return test
        return None

    def glob_ecus(self, globStr):  # type: (str) -> typing.List[Ecu]

        print("def : canmatrix - CanMatrix - glob_ecus")

        """
        Find ECUs by given glob pattern.

        :param globStr: glob pattern to filter BoardUnits. See `fnmatch.fnmatchcase`.
        :rtype: list of Ecu
        """
        return_array = []
        for test in self.ecus:
            if fnmatch.fnmatchcase(test.name, globStr):
                return_array.append(test)
        return return_array

    def add_frame(self, frame):  # type: (Frame) -> Frame

        print("def : canmatrix - CanMatrix - add_frame - [{}]".format(frame))

        """Add the Frame to the Matrix.

        :param Frame frame: Frame to add
        :return: the inserted Frame
        """
        self.frames.append(frame)
        return self.frames[len(self.frames) - 1]

    def add_signal(self, signal):  # type: (Signal) -> Signal

        # print("def : canmatrix - CanMatrix - add_signal - [{}]".format(signal))

        """Add the Frame to the Matrix.

        :param Frame frame: Frame to add
        :return: the inserted Frame
        """
        self.signals.append(signal)
        return self.signals[len(self.signals) - 1]

    def remove_frame(self, frame):  # type: (Frame) -> None

        print("def : canmatrix - CanMatrix - remove_frame")

        """Remove the Frame from Matrix.

        :param Frame frame: frame to remove from CAN Matrix
        """
        self.frames.remove(frame)

    def add_signal(self, signal):  # type: (Signal) -> Signal

        # print("def : canmatrix - CanMatrix - add_signal")

        """
        Add Signal to Frame.

        :param Signal signal: Signal to be added.
        :return: the signal added.
        """
        self.signals.append(signal)
        return self.signals[len(self.signals) - 1]

    def remove_signal(self, signal):  # type: (Signal) -> None

        print("def : canmatrix - CanMatrix - remove_signal")

        """Remove the Frame from Matrix.

        :param Signal signal: frame to remove from CAN Matrix
        """
        self.signals.remove(signal)

    def delete_zero_signals(self):  # type: () -> None

        print("def : canmatrix - CanMatrix - delete_zero_signals")

        """Delete all signals with zero bit width from all Frames."""
        for frame in self.frames:
            for signal in frame.signals:
                if 0 == signal.size:
                    frame.signals.remove(signal)

    def del_signal_attributes(self, unwanted_attributes):  # type: (typing.Sequence[str]) -> None

        print("def : canmatrix - CanMatrix - del_signal_attributes")

        """Delete Signal attributes from all Signals of all Frames.

        :param list of str unwanted_attributes: List of attributes to remove
        """
        for frame in self.frames:
            for signal in frame.signals:
                for attrib in unwanted_attributes:
                    signal.del_attribute(attrib)

    def del_frame_attributes(self, unwanted_attributes):  # type: (typing.Sequence[str]) -> None

        print("def : canmatrix - CanMatrix - del_frame_attributes")

        """Delete Frame attributes from all Frames.

        :param list of str unwanted_attributes: List of attributes to remove
        """
        for frame in self.frames:
            for attrib in unwanted_attributes:
                frame.del_attribute(attrib)

    def recalc_dlc(self, strategy):  # type: (str) -> None

        print("def : canmatrix - CanMatrix - recalc_dlc")

        """Recompute DLC of all Frames.

        :param str strategy: selected strategy, "max" or "force".
        """
        for frame in self.frames:
            if "max" == strategy:
                frame.calc_dlc()
            if "force" == strategy:
                maxBit = 0
                for sig in frame.signals:
                    if sig.get_startbit() + int(sig.size) > maxBit:
                        maxBit = sig.get_startbit() + int(sig.size)
                max_byte = (maxBit + 7) // 8
                if frame.is_pdu_container:
                    max_byte *= len(frame.pdus)
                    for pdu in self.pdus:
                        max_byte += pdu.size
                frame.size = max_byte

    def rename_ecu(self, ecu_or_name, new_name):  # type: (typing.Union[Ecu, str], str) -> None

        print("def : canmatrix - CanMatrix - rename_ecu")

        """Rename ECU in the Matrix. Update references in all Frames.

        :param str or Ecu ecu_or_name: old name or ECU instance
        :param str new_name: new name
        """
        ecu = ecu_or_name if isinstance(ecu_or_name, Ecu) else self.ecu_by_name(ecu_or_name)
        if ecu is None:
            return
        old_name = ecu.name
        ecu.name = new_name
        for frame in self.frames:
            if old_name in frame.transmitters:
                frame.transmitters.remove(old_name)
                frame.add_transmitter(new_name)
            for signal in frame.signals:
                if old_name in signal.receivers:
                    signal.receivers.remove(old_name)
                    signal.add_receiver(new_name)
            frame.update_receiver()

    def add_ecu(self, ecu):  # type(Ecu) -> None  # todo return Ecu?

        print("def : canmatrix - CanMatrix - add_ecu - {}".format(ecu))

        """Add new ECU to the Matrix. Do nothing if ecu with the same name already exists.

        :param Ecu ecu: ECU name to add
        """
        for bu in self.ecus:
            if bu.name.strip() == ecu.name:
                return
        self.ecus.append(ecu)

    def del_ecu(self, ecu_or_glob):  # type: (typing.Union[Ecu, str]) -> None

        print("def : canmatrix - CanMatrix - del_ecu")

        """Remove ECU from Matrix and all Frames.

        :param str or Ecu ecu_or_glob: ECU instance or glob pattern to remove from list
        """
        ecu_list = [ecu_or_glob] if isinstance(ecu_or_glob, Ecu) else self.glob_ecus(ecu_or_glob)

        for ecu in ecu_list:
            if ecu in self.ecus:
                self.ecus.remove(ecu)
                for frame in self.frames:
                    frame.del_transmitter(ecu.name)
                    for signal in frame.signals:
                        signal.del_receiver(ecu.name)

                    frame.update_receiver()

    def update_ecu_list(self):  # type: () -> None

        #print("def : canmatrix - CanMatrix - update_ecu_list - [{},{},{},{}]".format(self.ecus,self.frames,self.signals,self.attributes))

        """Check all Frames and add unknown ECUs to the Matrix ECU list."""
        for frame in self.frames:
            for transmit_ecu in frame.transmitters:
                print("def : canmatrix - CanMatrix - update_ecu_list - TRANSMIT ECU : {}".format(transmit_ecu))
                self.add_ecu(Ecu(transmit_ecu))
            frame.update_receiver()
            for signal in frame.signals:
                for receive_ecu in signal.receivers:
                    print("def : canmatrix - CanMatrix - update_ecu_list - RECEIVE ECU : {}".format(receive_ecu))
                    self.add_ecu(Ecu(receive_ecu))

    def rename_frame(self, frame_or_name, new_name):  # type: (typing.Union[Frame,str], str) -> None

        print("def : canmatrix - CanMatrix - rename_frame")

        """Rename Frame.

        :param Frame or str frame_or_name: Old Frame instance or name or part of the name with '*' at the beginning or the end.
        :param str new_name: new Frame name, suffix or prefix
        """
        old_name = frame_or_name.name if isinstance(frame_or_name, Frame) else frame_or_name
        for frame in self.frames:
            if old_name[-1] == '*':
                old_prefix_len = len(old_name) - 1
                if frame.name[:old_prefix_len] == old_name[:-1]:
                    frame.name = new_name + frame.name[old_prefix_len:]
            if old_name[0] == '*':
                old_suffix_len = len(old_name) - 1
                if frame.name[-old_suffix_len:] == old_name[1:]:
                    frame.name = frame.name[:-old_suffix_len] + new_name
            elif frame.name == old_name:
                frame.name = new_name

    def del_frame(self, frame_or_name):  # type: (typing.Union[Frame, str]) -> None

        print("def : canmatrix - CanMatrix - del_frame")

        """Delete Frame from Matrix.

        :param Frame or str frame_or_name: Frame or name to delete"""
        frame = frame_or_name if isinstance(frame_or_name, Frame) else self.frame_by_name(frame_or_name)
        if frame:
            self.frames.remove(frame)

    def rename_signal(self, signal_or_name, new_name):  # type: (typing.Union[Signal, str], str) -> None

        print("def : canmatrix - CanMatrix - rename_signal")

        """Rename Signal.

        :param Signal or str signal_or_name: Old Signal instance or name or part of the name with '*' at the beginning or the end.
        :param str new_name: new Signal name, suffix or prefix
        """
        old_name = signal_or_name.name if isinstance(signal_or_name, Signal) else signal_or_name
        for frame in self.frames:
            if old_name[-1] == '*':
                old_prefix_len = len(old_name) - 1
                for signal in frame.signals:
                    if signal.name[:old_prefix_len] == old_name[:-1]:
                        signal.name = new_name + signal.name[old_prefix_len:]
            elif old_name[0] == '*':
                old_suffix_len = len(old_name) - 1
                for signal in frame.signals:
                    if signal.name[-old_suffix_len:] == old_name[1:]:
                        signal.name = signal.name[:-old_suffix_len] + new_name
            else:
                signal_found = frame.signal_by_name(old_name)
                if signal_found:
                    signal_found.name = new_name

    def del_signal(self, signal):  # type: (typing.Union[Signal, str]) -> None

        print("def : canmatrix - CanMatrix - del_signal")

        """Delete Signal from Matrix and all Frames.

        :param Signal or str signal: Signal instance or glob pattern to be deleted"""
        if isinstance(signal, Signal):
            for frame in self.frames:
                if signal in frame.signals:
                    frame.signals.remove(signal)
        else:
            for frame in self.frames:
                signal_list = frame.glob_signals(signal)
                for sig in signal_list:
                    frame.signals.remove(sig)

    def add_signal_receiver(self, globFrame, globSignal, ecu):  # type: (str, str, str) -> None

        print("def : canmatrix - CanMatrix - add_signal_receiver")

        """Add Receiver to all Frames and Signals by glob pattern.

        :param str globFrame: glob pattern for Frame name.
        :param str globSignal: glob pattern for Signal name. Only signals under globFrame are filtered.
        :param str ecu: Receiver ECU name
        """
        frames = self.glob_frames(globFrame)
        for frame in frames:
            for signal in frame.glob_signals(globSignal):
                signal.add_receiver(ecu)
            frame.update_receiver()

    def del_signal_receiver(self, globFrame, globSignal, ecu):  # type: (str, str, str) -> None

        print("def : canmatrix - CanMatrix - del_signal_receiver")

        """Delete Receiver from all Frames by glob pattern.

        :param str globFrame: glob pattern for Frame name.
        :param str globSignal: glob pattern for Signal name. Only signals under globFrame are filtered.
        :param str ecu: Receiver ECU name
        """
        frames = self.glob_frames(globFrame)
        for frame in frames:
            for signal in frame.glob_signals(globSignal):
                signal.del_receiver(ecu)
            frame.update_receiver()

    def add_frame_transmitter(self, globFrame, ecu):  # type: (str, str) -> None

        print("def : canmatrix - CanMatrix - add_frame_transmitter")

        """Add Transmitter to all Frames by glob pattern.

        :param str globFrame: glob pattern for Frame name.
        :param str ecu: Receiver ECU name
        """
        frames = self.glob_frames(globFrame)
        for frame in frames:
            frame.add_transmitter(ecu)

    def add_frame_receiver(self, globFrame, ecu):  # type: (str, str) -> None

        print("def : canmatrix - CanMatrix - add_frame_receiver")

        """Add Receiver to all Frames by glob pattern.

        :param str globFrame: glob pattern for Frame name.
        :param str ecu: Receiver ECU name
        """
        frames = self.glob_frames(globFrame)
        for frame in frames:
            for signal in frame.signals:
                signal.add_receiver(ecu)

    def del_frame_transmitter(self, globFrame, ecu):  # type: (str, str) -> None

        print("def : canmatrix - CanMatrix - del_frame_transmitter")

        """Delete Transmitter from all Frames by glob pattern.

        :param str globFrame: glob pattern for Frame name.
        :param str ecu: Receiver ECU name
        """
        frames = self.glob_frames(globFrame)
        for frame in frames:
            frame.del_transmitter(ecu)

    def merge(self, mergeArray):  # type: (typing.Sequence[CanMatrix]) -> None

        print("def : canmatrix - CanMatrix - merge")

        """Merge multiple Matrices to this Matrix.

        Try to copy all Frames and all environment variables from source Matrices. Don't make duplicates.
        Log collisions.

        :param list of Matrix mergeArray: list of source CAN Matrices to be merged to to self.
        """
        for dbTemp in mergeArray:  # type: CanMatrix
            for frame in dbTemp.frames:
                copyResult = canmatrix.copy.copy_frame(frame.arbitration_id, dbTemp, self)
                if copyResult is False:
                    logger.error(
                        "ID Conflict, could not copy/merge frame " + frame.name + "  %xh " % frame.arbitration_id.id + self.frame_by_id(
                            frame.arbitration_id).name
                    )
            for envVar in dbTemp.env_vars:
                if envVar not in self.env_vars:
                    self.add_env_var(envVar, dbTemp.env_vars[envVar])
                else:
                    logger.error(
                        "Name Conflict, could not copy/merge EnvVar " + envVar)

    def encode(self, frame_id, data):  # type: (ArbitrationId, typing.Mapping[str, typing.Any]) -> bytes

        print("def : canmatrix - CanMatrix - encode")

        """Return a byte string containing the values from data packed
        according to the frame format.

        :param frame_id: frame id
        :param data: data dictionary
        :return: A byte string of the packed values.
        """
        return self.frame_by_id(frame_id).encode(data)

    def decode_pycan(self, pycan_msg):

        print("def : canmatrix - CanMatrix - decode_pycan")

        """Return OrderedDictionary with Signal Name: object decodedSignal

        :param pycan_msg: python-can message
        :return: OrderedDictionary
        """
        canmatrix_arbitration_id = canmatrix.ArbitrationId(pycan_msg.arbitration_id, extended=pycan_msg.is_extended_id)
        return self.decode(canmatrix_arbitration_id, pycan_msg.data)

    def decode(self, frame_id, data):  # type: (ArbitrationId, bytes) -> typing.Mapping[str, typing.Any]

        print("def : canmatrix - CanMatrix - decode")

        """Return OrderedDictionary with Signal Name: object decodedSignal

        :param frame_id: frame id
        :param data: Iterable or bytes.
            i.e. (0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8)
        :return: OrderedDictionary
        """
        if not self.contains_j1939:
            return self.frame_by_id(frame_id).decode(data)
        elif frame_id.extended:
            frame = self.frame_by_id(frame_id)
            if frame is None:
                frame = self.frame_by_pgn(frame_id.pgn)
            if frame:
                return frame.decode(data)
            else:
                return {}
        else:
            return {}

    def enum_attribs_to_values(self):  # type: () -> None

        #print("def : canmatrix - CanMatrix - enum_attribs_to_values")

        for define in self.ecu_defines:

            print("def : canmatrix - CanMatrix - enum_attribs_to_values - ECU DEFINE - {}".format(define))
            if self.ecu_defines[define].type == "ENUM":
                for bu in self.ecus:
                    if define in bu.attributes:
                        bu.attributes[define] = self.ecu_defines[define].values[int(bu.attributes[define])]
        for define in self.frame_defines:

            print("def : canmatrix - CanMatrix - enum_attribs_to_values - FRAME DEFINE - {}".format(define))
            if self.frame_defines[define].type == "ENUM":
                curr_frame = ''
                for signal in self.signals:
                    frame = signal.frames[0]
                    if define in frame.attributes and frame != curr_frame:
                        frame.attributes[define] = self.frame_defines[define].values[int(frame.attributes[define])]
                        curr_frame = frame

        for define in self.signal_defines:

            print("def : canmatrix - CanMatrix - enum_attribs_to_values - SIGNAL DEFINE - {}".format(define))
            if self.signal_defines[define].type == "ENUM":
                for frame in self.frames:
                    for signal in frame.signals:
                        if define in signal.attributes:
                            signal.attributes[define] = self.signal_defines[define].values[
                                int(signal.attributes[define])]

    def enum_attribs_to_keys(self):  # type: () -> None

        for define in self.ecu_defines:
            if self.ecu_defines[define].type == "ENUM":
                for bu in self.ecus:
                    print("def : canmatrix - CanMatrix - enum_attribs_to_keys - ECU - BU")
                    if define in bu.attributes:
                        if len(bu.attributes[define]) > 0:
                            bu.attributes[define] = self.ecu_defines[define].values.index(bu.attributes[define])
                            bu.attributes[define] = str(bu.attributes[define])
        for define in self.frame_defines:
            if self.frame_defines[define].type == "ENUM":
                print("def : canmatrix - CanMatrix - enum_attribs_to_keys - FRM IS ENUM - {}".format(self.frame_defines[define].type))
                for signal in self.signals:

                    frame = signal.frames
                    if define in frame.attributes:
                        print("def : canmatrix - CanMatrix - enum_attribs_to_keys - FRM - {}".format(define))
                        #if len(frame.attributes[define]) > 0:
                            #print("def : canmatrix - CanMatrix - enum_attribs_to_keys - FRM - {}".format(frame.attributes[define]))
                            #print("def : canmatrix - CanMatrix - enum_attribs_to_keys - FRM VALUE - ",type(self.frame_defines[define].values))
                            #frame.attributes[define] = self.frame_defines[define].values.index(frame.attributes[define])
                            #print("def : canmatrix - CanMatrix - enum_attribs_to_keys - FRM -END")
                            #frame.attributes[define] = str(frame.attributes[define])
                            #frame.attributes[define] = "0"

        for define in self.signal_defines:
            if self.signal_defines[define].type == "ENUM":
                for frame in self.frames:
                    for signal in frame.signals:
                        print("def : canmatrix - CanMatrix - enum_attribs_to_keys - SIG - ")
                        if define in signal.attributes:
                            signal.attributes[define] = self.signal_defines[define].values.index(
                                signal.attributes[define])
                            signal.attributes[define] = str(signal.attributes[define])
