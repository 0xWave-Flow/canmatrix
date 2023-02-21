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

#
# this script exports dbc-files from a canmatrix-object
# dbc-files are the can-matrix-definitions of the CANoe (Vector Informatic)

from __future__ import absolute_import, division, print_function

import collections
import copy
import decimal
import logging
import math
import re
import typing
from builtins import *

import canmatrix
import canmatrix.utils

logger = logging.getLogger(__name__)


def default_float_factory(value):  # type: (typing.Any) -> decimal.Decimal

    # print("def : format - dbc - default_float_factory - [{}]".format(value))

    return decimal.Decimal(value)


def normalize_name(name, whitespace_replacement):  # type: (str, str) -> str

    print("def : format - dbc - normalize_name")

    name = re.sub(r'\s+', whitespace_replacement, name)

    if ' ' in name:
        name = '"' + name + '"'

    return name


def format_float(f):  # type: (typing.Any) -> str

    #print("def : format - dbc - format_float")

    s = str(f).upper()
    if s.endswith('.0'):
        s = s[:-2]

    if 'E' in s:
        tmp = s.split('E')
        s = '%sE%s%s' % (tmp[0], tmp[1][0], tmp[1][1:].rjust(3, '0'))

    #print("def : format - dbc - format_float - [{}  ,  {}]".format(f,s.upper()))
    return s.upper()


def check_define(define):  # type: (canmatrix.Define) -> None

    #print("def : format - dbc - check_define - {}".format(define.type))

    # check if define is compatible with dbc. else replace by STRING
    if define.type not in ["ENUM", "STRING", "INT", "HEX", "FLOAT"]:
        logger.warning("dbc export of attribute type %s not supported; replaced by STRING", define.type)
        define.definition = "STRING"
        define.type = "STRING"


def create_define(data_type, define, define_type, defaults):

    #print("def : format - dbc - create_define")

    # type: (str, canmatrix.Define, str, typing.MutableMapping[str, str]) -> str
    check_define(define)
    define_string = "BA_DEF_ " + define_type
    define_string += ' "' + data_type + '" '
    define_string += define.definition + ';\n'

    if data_type not in defaults and define.defaultValue is not None:
        if define.type == "ENUM" or define.type == "STRING":
            defaults[data_type] = '"' + define.defaultValue + '"'
        else:
            defaults[data_type] = define.defaultValue

    print("def : format - dbc - create_define - {}".format(define_string))

    return define_string


def create_attribute_string(attribute, attribute_class, name, value, is_string):

    #print("def : format - dbc - create_attribute_string")

    # type: (str, str, str, typing.Any, bool) -> str
    if is_string:
        value = '"' + value + '"'
    elif not value:
        value = '""'
    attribute_string = 'BA_ "' + attribute + '" ' + attribute_class + ' ' + name + ' ' + str(value) + ';\n'

    print("def : format - dbc - create_attribute_string - {}".format(attribute_string))
    return attribute_string


def create_comment_string(comment_class, comment_ident, comment, export_encoding, export_comment_encoding,
                          ignore_encoding_errors):

    #print("def : format - dbc - create_comment_string")

    # type: (str, str, str, str, str, str) -> bytes
    if len(comment) == 0:
        return b""
    comment_string = ("CM_ " + comment_class + " " + comment_ident + ' "').encode(export_encoding, 'ignore')
    comment_string += comment.replace('"', '\\"').encode(export_comment_encoding, 'ignore')
    comment_string += '";\n'.encode(export_encoding, ignore_encoding_errors)

    print("def : format - dbc - create_comment_string - {}".format(comment_string))
    return comment_string


def dump(in_db, f, **options):

    print("def : format - dbc - dump")

    # type: (canmatrix.CanMatrix, typing.IO, **typing.Any) -> None
    # create copy because export changes database
    db = copy.deepcopy(in_db)
    dbc_export_encoding = options.get("dbcExportEncoding", 'iso-8859-1')
    dbc_export_comment_encoding = options.get("dbcExportCommentEncoding", dbc_export_encoding)
    compatibility = options.get('compatibility', True)
    ignore_encoding_errors = options.get("ignoreEncodingErrors", "ignore")
    write_val_table = options.get("writeValTable", True)

    whitespace_replacement = options.get("whitespaceReplacement", '_')
    if whitespace_replacement in ['', None] or {' ', '\t'}.intersection(whitespace_replacement):
        logger.warning(
            "Settings may result in whitespace in DBC variable names.  This is not supported by the DBC format.")

    if db.contains_fd and db.contains_j1939:
        db.add_frame_defines("VFrameFormat",
                             'ENUM  "StandardCAN","ExtendedCAN","reserved","J1939PG","reserved","reserved",'
                             '"reserved","reserved","reserved","reserved","reserved","reserved","reserved",'
                             '"reserved","StandardCAN_FD","ExtendedCAN_FD"')
        logger.warning("dbc export not fully compatible to candb, because both J1939 and CAN_FD frames are defined")

    elif db.contains_fd:
        db.add_global_defines("BusType", "STRING")
        db.add_attribute("BusType", "CAN FD")
        db.add_frame_defines("VFrameFormat",
                             'ENUM  "StandardCAN","ExtendedCAN","reserved","reserved","reserved","reserved",'
                             '"reserved","reserved","reserved","reserved","reserved","reserved","reserved",'
                             '"reserved","StandardCAN_FD","ExtendedCAN_FD"')

    elif db.contains_j1939:
        db.add_global_defines("ProtocolType", "STRING")
        db.add_attribute("ProtocolType", "J1939")
        db.add_frame_defines("VFrameFormat", 'ENUM  "StandardCAN","ExtendedCAN","reserved","J1939PG"')

    if db.contains_fd or db.contains_j1939:
        for signal in db.signals:
            frame = signal.frames
            if frame.is_fd:
                if frame.arbitration_id.extended:
                    frame.add_attribute("VFrameFormat", "ExtendedCAN_FD")
                else:
                    frame.add_attribute("VFrameFormat", "StandardCAN_FD")
            elif frame.is_j1939:
                frame.add_attribute("VFrameFormat", "J1939PG")
            else:
                if frame.arbitration_id.extended:
                    frame.add_attribute("VFrameFormat", "ExtendedCAN")
                else:
                    frame.add_attribute("VFrameFormat", "StandardCAN")

    # keys into enum values
    db.enum_attribs_to_keys()
    # shorten long environment variable names
    for env_var_name in copy.deepcopy(db.env_vars):
        if len(env_var_name) > 32:
            db.add_env_attribute(env_var_name, "SystemEnvVarLongSymbol", env_var_name)
            db.env_vars[env_var_name[:32]] = db.env_vars.pop(env_var_name)
            db.add_env_defines("SystemEnvVarLongSymbol", "STRING")
    # write dbc file headers
    NS_list = ['NS_DESC_', 'CM_', 'BA_DEF_', 'BA_', 'VAL_', 'CAT_DEF_', 'CAT_', 'FILTER', 'BA_DEF_DEF_',
               'EV_DATA_', 'ENVVAR_DATA_', 'SGTYPE_', 'SGTYPE_VAL_', 'BA_DEF_SGTYPE_', 'BA_SGTYPE_',
               'SIG_TYPE_REF_', 'VAL_TABLE_', 'SIG_GROUP_', 'SIG_VALTYPE_', 'SIGTYPE_VALTYPE_', 'BO_TX_BU_',
               'BA_DEF_REL_', 'BA_REL_', 'BA_DEF_DEF_REL_', 'BU_SG_REL_', 'BU_EV_REL_', 'BU_BO_REL_', 'SG_MUL_VAL_']

    header = "VERSION \"Excel 2 DBC\"\n\n\nNS_ :\n"
    for NS in NS_list:
        header += "\t" + NS + "\n"

    header += 'BS_:\n\n'
    print("def : format - dbc - dump - HEADER : {}".format(header.encode(dbc_export_encoding, ignore_encoding_errors)))
    f.write(header.encode(dbc_export_encoding, ignore_encoding_errors))

    # ECUs
    f.write("BU_: ".encode(dbc_export_encoding, ignore_encoding_errors))
    ecu: object
    for ecu in db.ecus:
        # fix long ecu names:
        if len(ecu.name) > 32 and ecu.name != 'Vector__XXX':
            ecu.add_attribute("SystemNodeLongSymbol", ecu.name)
            ecu.name = ecu.name[0:32]
            db.add_ecu_defines("SystemNodeLongSymbol", "STRING")
        elif ecu.name == 'Vector__XXX':
            ecu.name = ''

        print("def : format - dbc - dump - ECU : {}".format(ecu.name))
        f.write((ecu.name + " ").encode(dbc_export_encoding, ignore_encoding_errors))
    f.write("\n".encode(dbc_export_encoding, ignore_encoding_errors))

    # write value table VAL_TABLE_
    if write_val_table:
        # ValueTables
        for signal in db.signals:
            if signal.values:
                f.write(("VAL_TABLE_ " + signal.name).encode(dbc_export_encoding, ignore_encoding_errors))
                for key, val in signal.values.items():
                    print("def : format - dbc - dump - VALUE TABLE : {} , {}".format(str(key),val))
                    f.write(' {} "{}"'.format(str(key), val).encode(dbc_export_encoding, ignore_encoding_errors))
                f.write(";\n".encode(dbc_export_encoding, ignore_encoding_errors))
    f.write("\n".encode(dbc_export_encoding, ignore_encoding_errors))

    output_names = collections.defaultdict(dict)
    # type: typing.Dict[canmatrix.Frame, typing.Dict[canmatrix.Signal, str]]

    # eval signals/frames name and cycle time, add some defines

    for signal in db.signals:
        # fix long frame names
        if len(signal.frames.name) > 32:
            #signal.frames.add_attribute("SystemMessageLongSymbol", signal.frames.name)
            signal.frames.name = signal.frames.name[0:32]
            #db.add_frame_defines("SystemMessageLongSymbol", "STRING")

        # fix long signal names
        if len(signal.name) > 32:
            #signal.add_attribute("SystemSignalLongSymbol", signal.name)
            signal.name = signal.name[0:32]
            #db.add_signal_defines("SystemSignalLongSymbol", "STRING")

        if signal.frames.cycle_time != 0:
            pass
            #signal.frames.add_attribute("GenMsgCycleTime", signal.frames.cycle_time)

        if signal.cycle_time != 0:
            pass
            #signal.add_attribute("GenSigCycleTime", signal.cycle_time)
        if signal.phys2raw(None) != 0:
            pass
            #signal.add_attribute("GenSigStartValue", signal.phys2raw(None))

        name = signal.name

        if compatibility:
            name = re.sub("[^A-Za-z0-9]", whitespace_replacement, name)
            if name[0].isdigit():
                name = whitespace_replacement + name

        output_names[signal.frames][signal] = name

    # add signal/message cycle time and signal startvalue define
    if len(db.signals) > 0:
        if max([x.frames.cycle_time for x in db.signals]) > 0:
            pass
            #db.add_frame_defines("GenMsgCycleTime", 'INT 0 65535')
        if len([x.cycle_time for x in db.signals]) > 0:
            if max([x.cycle_time for x in db.signals]) > 0:
                pass
                #db.add_signal_defines("GenSigCycleTime", 'INT 0 65535')

            if max([x.phys2raw(None) for x in db.signals]) > 0 or min(
                    [y.phys2raw(None) for y in db.signals]) < 0:
                pass
                #db.add_signal_defines("GenSigStartValue", 'FLOAT 0 100000000000')

    # write message and signal loop
    curr_frame = None
    for signal in db.signals:
        frame = signal.frames
        # new frame detect
        if frame.name != curr_frame:
            f.write("\n".encode(dbc_export_encoding, ignore_encoding_errors))
            curr_frame = frame.name
            multiplex_written = False

            if len(frame.transmitters) == 0:
                frame.add_transmitter("Vector__XXX")

            # write BO_ (frame information) to dbc file
            print("def : format - dbc - dump - FRAME : {} , {} , {} , {}".format(
                frame.arbitration_id,
                frame.name,
                frame.size,
                frame.transmitters))

            f.write(
                ("BO_ %d " %
                 frame.arbitration_id.to_compound_integer() +
                 frame.name +
                 ": %d " %
                 frame.size +
                 frame.transmitters[0] +
                 "\n").encode(dbc_export_encoding, ignore_encoding_errors))

        if signal.multiplex == 'Multiplexor' and multiplex_written and not frame.is_complex_multiplexed:
            continue

        # create signal information in dbc format SG_
        signal_line = " SG_ " + output_names[frame][signal] + " "

        if signal.mux_val is not None:
            signal_line += "m{}".format(int(signal.mux_val))
            if signal.multiplex != 'Multiplexor':
                signal_line += " "

        if signal.multiplex == 'Multiplexor':
            signal_line += "M "
            multiplex_written = True

        # eval signal is_signed
        if signal.is_signed == 'Signed':
            sign = '-'
        else:
            sign = '+'

        signal_line += (": %d|%d@%d%c" %
                        (signal.msb,
                         signal.size,
                         signal.is_little_endian,
                         sign))
        signal_line += " (%s,%s)" % (format_float(signal.factor), format_float(signal.offset))
        signal_line += " [{}|{}]".format(format_float(signal.min), format_float(signal.max))
        signal_line += ' "'

        if signal.unit is None:
            signal.unit = ""


        signal_line += signal.unit
        signal_line += '" '

        if len(signal.receivers) == 0:
            signal.add_receiver('Vector__XXX')
        signal_line += ','.join(signal.receivers) + "\n"

        print("def : format - dbc - dump - SIGNAL : {}".format(signal.unit))


        print("def : format - dbc - dump - SIGNAL - BUG : {}",signal_line.encode(dbc_export_encoding, ignore_encoding_errors))

        f.write(signal_line.encode(dbc_export_encoding, ignore_encoding_errors))

    f.write("\n".encode(dbc_export_encoding, ignore_encoding_errors))

    # second Sender:
    for signal in db.signals:
        frame = signal.frames
        if len(frame.transmitters) > 1:
            f.write(("BO_TX_BU_ %d : %s;\n" % (
                frame.arbitration_id.to_compound_integer(), ','.join(frame.transmitters))).encode(
                dbc_export_encoding,
                ignore_encoding_errors))

    # frame comments CM_BO_
    # wow, there are dbcs where comments are encoded with other coding than rest of dbc...
    for signal in db.signals:
        frame = signal.frames
        f.write(create_comment_string("BO_", "%d " % frame.arbitration_id.to_compound_integer(), frame.comment,
                                      dbc_export_encoding, dbc_export_comment_encoding, dbc_export_encoding))
    f.write("\n".encode(dbc_export_encoding, ignore_encoding_errors))

    # signal comments CM_SG_
    for signal in db.signals:
        frame = signal.frames
        if signal.comment:
            name = output_names[frame][signal]
            f.write(create_comment_string(
                "SG_",
                "%d " % frame.arbitration_id.to_compound_integer() + name,
                signal.comment,
                dbc_export_encoding,
                dbc_export_comment_encoding, dbc_export_encoding))

    # ecu comments CM_BU_
    for ecu in db.ecus:
        if ecu.comment:
            f.write(create_comment_string("BU_", ecu.name, ecu.comment, dbc_export_encoding,
                                          dbc_export_comment_encoding, dbc_export_encoding))

    # write defines
    defaults = {}  # type: typing.Dict[str, str]
    for (data_type, define) in sorted(list(db.frame_defines.items())):
        print("def : format - dbc - dump - BO_ DEFINE : [{} , {}]".format(data_type,define))
        f.write(create_define(data_type, define, "BO_", defaults).encode(dbc_export_encoding, 'replace'))

    for (data_type, define) in sorted(list(db.signal_defines.items())):
        print("def : format - dbc - dump - SG_ DEFINE : [{} , {}]".format(data_type, define))
        f.write(create_define(data_type, define, "SG_", defaults).encode(dbc_export_encoding, 'replace'))

    for (data_type, define) in sorted(list(db.ecu_defines.items())):
        print("def : format - dbc - dump - BU_ DEFINE : [{} , {}]".format(data_type, define))
        f.write(create_define(data_type, define, "BU_", defaults).encode(dbc_export_encoding, 'replace'))

    for (data_type, define) in sorted(list(db.env_defines.items())):
        print("def : format - dbc - dump - EV_ DEFINE : [{} , {}]".format(data_type, define))
        f.write(create_define(data_type, define, "EV_", defaults).encode(dbc_export_encoding, 'replace'))

    for (data_type, define) in sorted(list(db.global_defines.items())):
        f.write(create_define(data_type, define, "", defaults).encode(dbc_export_encoding, 'replace'))

    for define_name in sorted(defaults):

        print("def : format - dbc - dump - DEFINE : {}".format(define_name))

        f.write(('BA_DEF_DEF_ "' + define_name + '" ').encode(dbc_export_encoding, ignore_encoding_errors) +
                defaults[define_name].encode(dbc_export_encoding, 'replace') + ';\n'.encode(dbc_export_encoding,
                                                                                            ignore_encoding_errors))
    # ecu-attributes:
    for ecu in db.ecus:
        for attrib, val in sorted(ecu.attributes.items()):

            print("def : format - dbc - dump - ECU ATTRIBUTE : [{} , {}]".format(attrib, val))

            f.write(
                create_attribute_string(attrib, "BU_", ecu.name, val, db.ecu_defines[attrib].type == "STRING").encode(
                    dbc_export_encoding, ignore_encoding_errors))

    # global-attributes:
    for attrib, val in sorted(db.attributes.items()):

        print("def : format - dbc - dump - GLOBAL ATTRIBUTE : [{} , {}]".format(attrib, val))

        f.write(create_attribute_string(attrib, "", "", val, db.global_defines[attrib].type == "STRING").encode(
            dbc_export_encoding, ignore_encoding_errors))

    # messages-attributes:
    curr_frame = None
    for signal in db.signals:
        frame = signal.frames
        if frame.name != curr_frame:
            for attrib, val in sorted(frame.attributes.items()):

                print("def : format - dbc - dump - MESSAGE ATTRIBUTE : [{} , {}]".format(attrib,val))

                f.write(create_attribute_string(attrib, "BO_", str(frame.arbitration_id.to_compound_integer()), val,
                                                db.frame_defines[attrib].type == "STRING").encode(dbc_export_encoding,
                                                                                                  ignore_encoding_errors))
            curr_frame = frame.name

    # signal-attributes:
    for signal in db.signals:
        frame = signal.frames
        for attrib, val in sorted(signal.attributes.items()):
            name = output_names[frame][signal]
            if isinstance(val, float):
                val = format_float(val)
            if attrib in db.signal_defines:

                print("def : format - dbc - dump - SIGNAL ATTRIBUTE : {}".format(attrib))

                f.write(create_attribute_string(
                    attrib, "SG_", '%d ' % frame.arbitration_id.to_compound_integer() + name, val,
                                   db.signal_defines[attrib].type == "STRING").encode(dbc_export_encoding,
                                                                                      ignore_encoding_errors))
    # environment-attributes:
    for env_var_name, env_var in db.env_vars.items():
        if "attributes" in env_var:
            for attribute, value in env_var["attributes"].items():

                print("def : format - dbc - dump - ENVIRONMENT ATTRIBUTE : [{} , {}]".format(attrib, val))

                f.write(create_attribute_string(attribute, "EV_", "", value,
                                                db.env_defines[attribute].type == "STRING")
                        .encode(dbc_export_encoding, ignore_encoding_errors))

    # signal-values: value table
    multiplex_written = False
    for signal in db.signals:
        if signal.multiplex == 'Multiplexor' and multiplex_written:
            continue

        multiplex_written = True

        if signal.values:
            f.write(
                ('VAL_ %d ' %
                 signal.frames.arbitration_id.to_compound_integer() +
                 output_names[signal.frames][signal]).encode(dbc_export_encoding, ignore_encoding_errors))
            for attr_name, val in sorted(signal.values.items(), key=lambda x: int(x[0])):
                if '"' in val:
                    val = val.replace('"', '\\"')
                f.write(
                    (' ' + str(attr_name) + ' "' + val + '"').encode(dbc_export_encoding, ignore_encoding_errors))

            f.write(";\n".encode(dbc_export_encoding, ignore_encoding_errors))

    # SIG_VALTYPE
    for signal in db.signals:
        if signal.is_float:
            if int(signal.size) > 32:
                f.write(('SIG_VALTYPE_ %d %s : 2;\n' % (
                    signal.frames.arbitration_id.to_compound_integer(), output_names[signal.frames][signal])).encode(
                    dbc_export_encoding, ignore_encoding_errors))
            else:
                f.write(('SIG_VALTYPE_ %d %s : 1;\n' % (
                    signal.frames.arbitration_id.to_compound_integer(), output_names[signal.framesme][signal])).encode(
                    dbc_export_encoding, ignore_encoding_errors))

    # signal-groups:
    for signal in db.signals:
        frame = signal.frames
        for sigGroup in frame.signalGroups:
            f.write(("SIG_GROUP_ " + str(frame.arbitration_id.to_compound_integer()) + " " + sigGroup.name +
                     " " + str(sigGroup.id) + " :").encode(dbc_export_encoding, ignore_encoding_errors))
            for s in sigGroup.signals:
                f.write((" " + output_names[frame][s]).encode(dbc_export_encoding, ignore_encoding_errors))
            f.write(";\n".encode(dbc_export_encoding, ignore_encoding_errors))

    for signal in db.signals:
        frame = signal.frames
        if frame.is_complex_multiplexed:
            if signal.muxer_for_signal is not None:
                f.write(("SG_MUL_VAL_ %d %s %s " % (
                    frame.arbitration_id.to_compound_integer(), signal.name, signal.muxer_for_signal)).encode(
                    dbc_export_encoding, ignore_encoding_errors))
                f.write((", ".join(["%d-%d" % (a, b) for a, b in signal.mux_val_grp])).encode(dbc_export_encoding,
                                                                                              ignore_encoding_errors))

                f.write(";\n".encode(dbc_export_encoding, ignore_encoding_errors))

    for env_var_name in db.env_vars:
        env_var = db.env_vars[env_var_name]
        f.write(("EV_ {0} : {1} [{2}|{3}] \"{4}\" {5} {6} {7} {8};\n".format(
            env_var_name, env_var["varType"], env_var["min"],
            env_var["max"], env_var["unit"], env_var["initialValue"],
            env_var["evId"], env_var["accessType"],
            ",".join(env_var["accessNodes"]))).encode(dbc_export_encoding, ignore_encoding_errors))


class _FollowUps(object):
    NOTHING, SIGNAL_COMMENT, FRAME_COMMENT, BOARD_UNIT_COMMENT, GLOBAL_COMMENT = range(5)


def load(f, **options):  # type: (typing.IO, **typing.Any) -> canmatrix.CanMatrix

    #print("def : format - dbc - load")

    dbc_import_encoding = options.get("dbcImportEncoding", 'UTF-8')
    dbc_comment_encoding = options.get("dbcImportCommentEncoding", dbc_import_encoding)
    float_factory = options.get('float_factory', default_float_factory)
    i = 0

    follow_up = _FollowUps.NOTHING
    comment = ""
    signal = None  # type: typing.Optional[canmatrix.Signal]
    frame = None
    board_unit = None
    db = canmatrix.CanMatrix()
    frames_by_id = {}  # type: typing.Dict[int, canmatrix.Frame]

    def hash_arbitration_id(arbitration_id):  # type: (canmatrix.ArbitrationId) -> int
        return hash((arbitration_id.id, arbitration_id.extended))

    def get_frame_by_id(arbitration_id):  # type: (canmatrix.ArbitrationId) -> typing.Optional[canmatrix.Frame]
        try:
            return frames_by_id[hash_arbitration_id(arbitration_id)]
        except KeyError:
            return None

    def add_frame_by_id(new_frame):  # type: (canmatrix.Frame) -> None
        frames_by_id[hash_arbitration_id(new_frame.arbitration_id)] = new_frame

    # read line by line
    for line in f:

        # print("def : format - dbc - load - LINE : {}".format(line))

        i = i + 1
        l = line.strip()
        if len(l) == 0:
            continue
        try:
            if follow_up == _FollowUps.SIGNAL_COMMENT:
                try:
                    comment += "\n" + l.decode(dbc_comment_encoding).replace('\\"', '"')
                except:
                    logger.error("Error decoding line: %d (%s)" % (i, line))
                if re.match(r'.*" *;\Z', l.decode(dbc_import_encoding).strip()) is not None:
                    follow_up = _FollowUps.NOTHING
                    if signal is not None:
                        signal.add_comment(comment[:-1].strip()[:-1])
                continue

            elif follow_up == _FollowUps.FRAME_COMMENT:
                try:
                    comment += "\n" + l.decode(dbc_comment_encoding).replace('\\"', '"')
                except:
                    logger.error("Error decoding line: %d (%s)" % (i, line))
                if re.match(r'.*" *;\Z', l.decode(dbc_import_encoding).strip()) is not None:
                    follow_up = _FollowUps.NOTHING
                    if frame is not None:
                        frame.add_comment(comment[:-1].strip()[:-1])
                continue

            elif follow_up == _FollowUps.BOARD_UNIT_COMMENT:
                try:
                    comment += "\n" + \
                               l.decode(dbc_comment_encoding).replace('\\"', '"')
                except:
                    logger.error("Error decoding line: %d (%s)" % (i, line))
                if re.match(r'.*" *;\Z', l.decode(dbc_import_encoding).strip()) is not None:
                    follow_up = _FollowUps.NOTHING
                    if board_unit is not None:
                        board_unit.add_comment(comment[:-1].strip()[:-1])
                continue

            # decoded dbc file
            decoded = l.decode(dbc_import_encoding).strip()

            # print("LINE OF *.DBC : " + decoded)
            # print("def : format - dbc - load - LINE DECODED : {}".format(decoded))

            # decoded message
            if decoded.startswith("BO_ "):

                print("def : format - dbc - load - BO_ : {}".format(decoded))

                regexp = re.compile(r"^BO_ ([^\ ]+) ([^\ ]+) *: ([^\ ]+) ([^\ ]+)")
                temp = regexp.match(decoded)

                #print("def : format - dbc - load - BO_ SPLIT : {} - {} - {} - {} - {}".format(temp.group(0),temp.group(1),temp.group(2),temp.group(3),temp.group(4)))

                frame = canmatrix.Frame(temp.group(2), arbitration_id=int(temp.group(1)),
                                    size=int(temp.group(3)), transmitters=temp.group(4).split())

                if int(temp.group(1)) >= 0x80000000:
                    frame.arbitration_id.id = (int(temp.group(1)) - 0x80000000)
                    frame.arbitration_id.extended = True

                print("FIND FRAME : {} , {} , {} , {}".format(frame.name,frame.arbitration_id,temp.group(1),int(temp.group(1))))
                print("def : format - dbc - load - BO_ FRAME : {}".format(frame))
                add_frame_by_id(frame)

                if frame.name == "VECTOR__INDEPENDENT_SIG_MSG" and frame.arbitration_id.id == 0:
                    frame.arbitration_id.id = 3221225472

            # decoded signal
            elif decoded.startswith("SG_ "):

                print("def : format - dbc - load - SG_ : {}".format(decoded))

                original_line = l
                if decoded.strip().endswith(r'"'):
                    decoded += r" Vector__XXX"
                    original_line += b" Vector__XXX"

                pattern = r"^SG_ +(\w+) *: *(\d+)\|(\d+)@(\d+)([\+|\-]) *\(([0-9.+\-eE]+), *([0-9.+\-eE]+)\) *\[([" \
                          r"0-9.+\-eE]+)\|([0-9.+\-eE]+)\] +\"(.*)\" +(.*)"

                regexp = re.compile(pattern)
                temp = regexp.match(decoded)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp_raw = regexp_raw.match(original_line)
                if temp:
                    receiver = [b.strip() for b in temp.group(11).split(',')]

                    extras = {}  # type: typing.Dict[typing.Any, typing.Any]

                    temp_signal = canmatrix.Signal(
                        temp.group(1),
                        msb=int(temp.group(2)),
                        size=int(temp.group(3)),
                        is_little_endian=(int(temp.group(4)) == 1),
                        is_signed=(temp.group(5) == '-'),
                        factor=temp.group(6),
                        offset=temp.group(7),
                        min=temp.group(8),
                        max=temp.group(9),
                        unit=temp_raw.group(10).decode(dbc_import_encoding),
                        receivers=receiver,
                        frames=list(),
                        **extras
                    )
                    temp_signal.add_frame(frame)

                    # print("FIND SIGNAL : " + temp_signal)
                    print("def : format - dbc - load - SG_ IS SIGNED: {} - {}".format(temp_signal.name,temp_signal.is_signed))

                    db.add_signal(temp_signal)

                # signal with multiplex
                else:

                    print("MULTIPLEX >>>>>")

                    pattern = r"^SG_ +(.+?) +(.+?) *: *(\d+)\|(\d+)@(\d+)([\+|\-]) *\(([0-9.+\-eE]+),([0-9.+\-eE]+)\) " \
                              r"*\[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] +\"(.*)\" +(.*)"
                    regexp = re.compile(pattern)
                    regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                    temp = regexp.match(decoded)
                    temp_raw = regexp_raw.match(original_line)
                    receiver = [b.strip() for b in temp.group(12).split(',')]
                    multiplex = temp.group(2)  # type: str

                    is_complex_multiplexed = False

                    if multiplex == 'M':
                        multiplex = 'Multiplexor'
                    elif multiplex.endswith('M'):
                        is_complex_multiplexed = True
                        multiplex = multiplex[:-1]

                    if multiplex != 'Multiplexor':
                        try:
                            multiplex = int(multiplex[1:])
                        except:
                            raise Exception('error decoding line', line)

                    extras = {}

                    temp_signal = canmatrix.Signal(
                        temp.group(1),
                        msb=int(temp.group(3)),
                        size=int(temp.group(4)),
                        is_little_endian=(int(temp.group(5)) == 1),
                        is_signed=(temp.group(6) == '-'),
                        factor=temp.group(7),
                        offset=temp.group(8),
                        min=temp.group(9),
                        max=temp.group(10),
                        unit=temp_raw.group(11).decode(dbc_import_encoding),
                        receivers=receiver,
                        multiplex=multiplex,
                        **extras
                    )

                    if is_complex_multiplexed:
                        temp_signal.is_multiplexer = True
                        temp_signal.multiplex = 'Multiplexor'
                        frame.is_complex_multiplexed = True
                    temp_signal.add_frame(frame)

            # decode other define or comment
            elif decoded.startswith("BO_TX_BU_ "):

                print("def : format - dbc - load - BO_TX_BU_ : {}".format(decoded))

                regexp = re.compile(r"^BO_TX_BU_ ([0-9]+) *: *(.+) *;")
                temp = regexp.match(decoded)
                frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(1))))
                for ecu_name in temp.group(2).split(','):
                    # print(">>>>> " + ecu_name)
                    frame.add_transmitter(ecu_name)

            # decode signal comment
            elif decoded.startswith("CM_ SG_ "):

                print("def : format - dbc - load - CM_ SG_ : {}".format(decoded))

                pattern = r"^CM_ +SG_ +(\w+) +(\w+) +\"(.*)\" *;"
                regexp = re.compile(pattern)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp = regexp.match(decoded)
                temp_raw = regexp_raw.match(l)
                if temp:
                    frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(1))))
                    signal = frame.signal_by_name(temp.group(2))
                    if signal:
                        try:
                            signal.add_comment(temp_raw.group(3).decode(
                                dbc_comment_encoding).replace('\\"', '"'))
                        except:
                            logger.error(
                                "Error decoding line: %d (%s)" %
                                (i, line))
                else:
                    pattern = r"^CM_ +SG_ +(\w+) +(\w+) +\"(.*)"
                    regexp = re.compile(pattern)
                    regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                    temp = regexp.match(decoded)
                    temp_raw = regexp_raw.match(l)
                    if temp:
                        frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(1))))
                        signal = frame.signal_by_name(temp.group(2))
                        try:
                            comment = temp_raw.group(3).decode(
                                dbc_comment_encoding).replace('\\"', '"')
                        except:
                            logger.error(
                                "Error decoding line: %d (%s)" %
                                (i, line))
                        follow_up = _FollowUps.SIGNAL_COMMENT

            # decode message comment
            elif decoded.startswith("CM_ BO_ "):

                print("def : format - dbc - load - CM_ BO_ : {}".format(decoded))

                pattern = r"^CM_ +BO_ +(\w+) +\"(.*)\" *;"
                regexp = re.compile(pattern)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp = regexp.match(decoded)
                temp_raw = regexp_raw.match(l)
                if temp:
                    frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(1))))
                    if frame:
                        try:
                            frame.add_comment(temp_raw.group(2).decode(
                                dbc_comment_encoding).replace('\\"', '"'))
                        except:
                            logger.error(
                                "Error decoding line: %d (%s)" %
                                (i, line))
                else:
                    pattern = r"^CM_ +BO_ +(\w+) +\"(.*)"
                    regexp = re.compile(pattern)
                    regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                    temp = regexp.match(decoded)
                    temp_raw = regexp_raw.match(l)
                    if temp:
                        frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(1))))
                        try:
                            comment = temp_raw.group(2).decode(
                                dbc_comment_encoding).replace('\\"', '"')
                        except:
                            logger.error(
                                "Error decoding line: %d (%s)" %
                                (i, line))
                        follow_up = _FollowUps.FRAME_COMMENT

            # decode ecu comment
            elif decoded.startswith("CM_ BU_ "):

                print("def : format - dbc - load - CM_ BU_ : {}".format(decoded))

                pattern = r"^CM_ +BU_ +(\w+) +\"(.*)\" *;"
                regexp = re.compile(pattern)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp = regexp.match(decoded)
                temp_raw = regexp_raw.match(l)
                if temp:
                    board_unit = db.ecu_by_name(temp.group(1))
                    if board_unit:
                        try:
                            board_unit.add_comment(temp_raw.group(2).decode(
                                dbc_comment_encoding).replace('\\"', '"'))
                        except:
                            logger.error("Error decoding line: %d (%s)" % (i, line))
                else:
                    pattern = r"^CM_ +BU_ +(\w+) +\"(.*)"
                    regexp = re.compile(pattern)
                    regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                    temp = regexp.match(decoded)
                    temp_raw = regexp_raw.match(l)
                    if temp:
                        board_unit = db.ecu_by_name(temp.group(1))
                        if board_unit:
                            try:
                                comment = temp_raw.group(2).decode(
                                    dbc_comment_encoding).replace('\\"', '"')
                            except:
                                logger.error(
                                    "Error decoding line: %d (%s)" %
                                    (i, line))
                            follow_up = _FollowUps.BOARD_UNIT_COMMENT

            # decode ecus
            elif decoded.startswith("BU_:"):

                print("def : format - dbc - load - BU_ : {}".format(decoded))

                pattern = r"^BU_\:(.*)"
                regexp = re.compile(pattern)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp = regexp.match(decoded)
                if temp:
                    my_temp_list = temp.group(1).split(' ')
                    for ele in my_temp_list:
                        if len(ele.strip()) > 1:
                            db.ecus.append(canmatrix.Ecu(ele))

            # decode values
            elif decoded.startswith("VAL_ "):

                print("def : format - dbc - load - VAL_ : {}".format(decoded))

                regexp = re.compile(r"^VAL_ +(\w+) +(\w+) +(.*) *;")
                temp = regexp.match(decoded)
                if temp:
                    signal_name = temp.group(2)
                    temp_list = list(canmatrix.utils.escape_aware_split(temp.group(3), '"'))

                    if signal_name:  # value for Frame
                        try:
                            for signal in db.signals:
                                if signal.name == signal_name:
                                    for i in range(math.floor(len(temp_list) / 2)):
                                        val = temp_list[i * 2 + 1]
                                        val = val.replace('\\"', '"')

                                        signal.add_values(temp_list[i * 2], val)
                        except:
                            logger.error("Error with Line: " + str(temp_list))
                    else:
                        logger.info("Warning: environment variables currently not supported")

            # decode value table
            elif decoded.startswith("VAL_TABLE_ "):

                print("def : format - dbc - load - VAL_TABLE_ : {}".format(decoded))

                regexp = re.compile(r"^VAL_TABLE_ +(\w+) +(.*) *;")
                temp = regexp.match(decoded)
                if temp:
                    table_name = temp.group(1)
                    temp_list = temp.group(2).split('"')
                    value_hash = {}
                    try:
                        for i in range(math.floor(len(temp_list) / 2)):
                            val = temp_list[i * 2 + 1]
                            value_hash[temp_list[i * 2].strip()] = val.strip()
                    except:
                        logger.error("Error with Line: " + str(temp_list))
                    db.add_value_table(table_name, value_hash)
                else:
                    logger.debug(l)

            # decode definitions
            elif decoded.startswith("BA_DEF_") and decoded[7:].strip()[:3] in ["SG_", "BO_", "BU_", "EV_"]:

                print("def : format - dbc - load - BA_DEF_ : {}".format(decoded))

                substring = decoded[7:].strip()
                define_type = substring[:3]
                substring = substring[3:].strip()
                pattern = r"^\"(.+?)\" +(.+); *"
                regexp = re.compile(pattern)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp = regexp.match(substring)
                substring_line = l[7:].strip()[3:].strip()
                temp_raw = regexp_raw.match(substring_line)
                if temp:
                    if define_type == "SG_":
                        db.add_signal_defines(temp.group(1), temp_raw.group(2).decode(dbc_import_encoding))
                    elif define_type == "BO_":
                        db.add_frame_defines(temp.group(1), temp_raw.group(2).decode(dbc_import_encoding))
                    elif define_type == "BU_":
                        db.add_ecu_defines(temp.group(1), temp_raw.group(2).decode(dbc_import_encoding))
                    elif define_type == "EV_":
                        db.add_env_defines(temp.group(1), temp_raw.group(2).decode(dbc_import_encoding))

            # decode define
            elif decoded.startswith("BA_DEF_ "):

                print("def : format - dbc - load - BA_DEF_ : {}".format(decoded))

                pattern = r"^BA_DEF_ +\"(.+?)\" +(.+) *;"
                regexp = re.compile(pattern)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp = regexp.match(decoded)
                temp_raw = regexp_raw.match(l)
                if temp:
                    db.add_global_defines(temp.group(1),
                                          temp_raw.group(2).decode(dbc_import_encoding))

            # decode attribute value definitions
            elif decoded.startswith("BA_ "):

                print("def : format - dbc - load - BA_ : {}".format(decoded))

                regexp = re.compile(r"^BA_ +\".+?\" +(.+)")
                tempba = regexp.match(decoded)

                if tempba.group(1).strip().startswith("BO_ "):
                    regexp = re.compile(r"^BA_ +\"(.+?)\" +BO_ +(\d+) +(.+) *; *")
                    temp = regexp.match(decoded)
                    get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(2)))).add_attribute(
                        temp.group(1), temp.group(3))
                elif tempba.group(1).strip().startswith("SG_ "):
                    regexp = re.compile(r"^BA_ +\"(.+?)\" +SG_ +(\d+) +(\w+) +(.+) *; *")
                    temp = regexp.match(decoded)
                    if temp is not None:
                        get_frame_by_id(
                            canmatrix.ArbitrationId.from_compound_integer(int(temp.group(2)))).signal_by_name(
                            temp.group(3)).add_attribute(temp.group(1), temp.group(4))
                elif tempba.group(1).strip().startswith("EV_ "):
                    regexp = re.compile(r"^BA_ +\"(.+?)\" +EV_ +(\w+) +(.*) *; *")
                    temp = regexp.match(decoded)
                    if temp is not None:
                        db.add_env_attribute(temp.group(2), temp.group(1), temp.group(3))
                elif tempba.group(1).strip().startswith("BU_ "):
                    regexp = re.compile(r"^BA_ +\"(.*?)\" +BU_ +(\w+) +(.+) *; *")
                    temp = regexp.match(decoded)
                    db.ecu_by_name(
                        temp.group(2)).add_attribute(
                        temp.group(1),
                        temp.group(3))
                else:
                    regexp = re.compile(
                        r"^BA_ +\"([A-Za-z0-9\-_]+)\" +([\"\w\-\.]+) *; *")
                    temp = regexp.match(decoded)
                    if temp:
                        db.add_attribute(temp.group(1), temp.group(2))

            # decode signal group
            elif decoded.startswith("SIG_GROUP_ "):

                print("def : format - dbc - load - SIG_GROUP_ : {}".format(decoded))

                regexp = re.compile(r"^SIG_GROUP_ +(\w+) +(\w+) +(\w+) +\:(.*) *; *")
                temp = regexp.match(decoded)
                frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(1))))
                if frame is not None:
                    signal_array = temp.group(4).split(' ')
                    frame.add_signal_group(temp.group(2), temp.group(3),
                                           signal_array)  # todo wrong annotation in canmatrix? Id is a string?

            # decode signal value type
            elif decoded.startswith("SIG_VALTYPE_ "):

                print("def : format - dbc - load - SIG_VALTYPE_ : {}".format(decoded))

                regexp = re.compile(r"^SIG_VALTYPE_ +(\w+) +(\w+)\s*\:(.*) *; *")
                temp = regexp.match(decoded)
                frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(temp.group(1))))
                if frame:
                    signal = frame.signal_by_name(temp.group(2))
                    signal.is_float = True

            # decode attribute default value definitions
            elif decoded.startswith("BA_DEF_DEF_ "):

                print("def : format - dbc - load - BA_DEF_DEF_ : {}".format(decoded))

                pattern = r"^BA_DEF_DEF_ +\"(.+?)\" +(.+?) *;"
                regexp = re.compile(pattern)
                regexp_raw = re.compile(pattern.encode(dbc_import_encoding))
                temp = regexp.match(decoded)
                temp_raw = regexp_raw.match(l)
                if temp:
                    db.add_define_default(temp.group(1),
                                          temp_raw.group(2).decode(dbc_import_encoding))

            # decode signal multiplex value
            elif decoded.startswith("SG_MUL_VAL_ "):

                print("def : format - dbc - load - SG_MUL_VAL_ : {}".format(decoded))

                pattern = r"^SG_MUL_VAL_ +([0-9]+) +([\w\-]+) +([\w\-]+) +(.*) *; *"
                regexp = re.compile(pattern)
                temp = regexp.match(decoded)
                if temp:
                    frame_id = temp.group(1)
                    signal_name = temp.group(2)
                    muxer_for_signal = temp.group(3)
                    mux_val_groups = temp.group(4).split(',')
                    frame = get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(int(frame_id)))
                    if frame is not None:
                        signal = frame.signal_by_name(signal_name)
                        frame.is_complex_multiplexed = True
                        signal.muxer_for_signal = muxer_for_signal
                        for muxVal in mux_val_groups:
                            mux_val_min, mux_val_max = muxVal.split("-")
                            mux_val_min_number = int(mux_val_min)
                            mux_val_max_number = int(mux_val_max)
                            signal.mux_val_grp.append([mux_val_min_number, mux_val_max_number])

            # decode environment value
            elif decoded.startswith("EV_ "):

                print("def : format - dbc - load - EV_ : {}".format(decoded))

                pattern = r"^EV_ +([\w\-\_]+?) *\: +([0-9]+) +\[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] +\"(.*?)\" +([" \
                          r"0-9.+\-eE]+) +([0-9.+\-eE]+) +([\w\-]+?) +(.*); *"
                regexp = re.compile(pattern)
                temp = regexp.match(decoded)

                var_name = temp.group(1)
                var_type = temp.group(2)
                min_value = temp.group(3)
                max_value = temp.group(4)
                unit = temp.group(5)
                initial_value = temp.group(6)
                ev_id = temp.group(7)
                access_type = temp.group(8)
                access_nodes = temp.group(9).split(",")
                db.add_env_var(var_name, {"varType": var_type, "min": min_value, "max": max_value, "unit": unit,
                                          "initialValue": initial_value, "evId": ev_id, "accessType": access_type,
                                          "accessNodes": access_nodes})

        # else:
        except:
            print("error with line no: %d" % i)
            print(line)

    # Backtracking
    env_var_names = list(db.env_vars.keys())
    for env_var_name in env_var_names:
        env_var = db.env_vars[env_var_name]
        if 'SystemEnvVarLongSymbol' in env_var.get("attributes", ""):
            long_name = env_var["attributes"]["SystemEnvVarLongSymbol"][1:-1]
            del (env_var["attributes"]["SystemEnvVarLongSymbol"])
            db.env_vars[long_name] = db.env_vars.pop(env_var_name)
    for ecu in db.ecus:
        if ecu.attributes.get("SystemNodeLongSymbol", None) is not None:
            ecu.name = ecu.attributes.get("SystemNodeLongSymbol")[1:-1]
            ecu.del_attribute("SystemNodeLongSymbol")

    for signal in db.signals:
        frame = signal.frames[0]
        frame.cycle_time = int(float(frame.attributes.get("GenMsgCycleTime", 0)))

        if frame.attributes.get("SystemMessageLongSymbol", None) is not None:
            frame.name = frame.attributes.get("SystemMessageLongSymbol")[1:-1]
            frame.del_attribute("SystemMessageLongSymbol")
        # frame.update_receiver()

        gen_sig_start_value = float_factory(signal.attributes.get("GenSigStartValue", "0"))
        signal.initial_value = (gen_sig_start_value * signal.factor) + signal.offset
        signal.cycle_time = int(signal.attributes.get("GenSigCycleTime", 0))
        if signal.attribute("SystemSignalLongSymbol") is not None:
            signal.name = signal.attribute("SystemSignalLongSymbol")[1:-1]
            signal.del_attribute("SystemSignalLongSymbol")

    for define in db.global_defines:
        if db.global_defines[define].type == "STRING":
            if define in db.attributes:
                db.attributes[define] = db.attributes[define][1:-1]
    for define in db.ecu_defines:
        if db.ecu_defines[define].type == "STRING":
            for ecu in db.ecus:
                if define in ecu.attributes:
                    ecu.attributes[define] = ecu.attributes[define][1:-1]
    for define in db.frame_defines:
        if db.frame_defines[define].type == "STRING":
            for signal in db.signals:
                frame = signal.frames[0]
                if define in frame.attributes:
                    frame.attributes[define] = frame.attributes[define][1:-1]
    for define in db.signal_defines:
        if db.signal_defines[define].type == "STRING":
            for signal in db.signals:
                if define in signal.attributes:
                    signal.attributes[define] = signal.attributes[define][1:-1]

    db.enum_attribs_to_values()
    for signal in db.signals:
        frame = signal.frames[0]
        if "_FD" in frame.attributes.get("VFrameFormat", ""):
            frame.is_fd = True
        if "J1939PG" in frame.attributes.get("VFrameFormat", ""):
            frame.is_j1939 = True
    db.update_ecu_list()

    return db
