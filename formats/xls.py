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
# this script exports xls-files from a canmatrix-object
# xls-files are the can-matrix-definitions displayed in Excel

from __future__ import absolute_import, division, print_function

import decimal
from decimal import *
import logging
import typing
from builtins import *

import past.builtins
import xlrd
import xlwt

import canmatrix
import canmatrix.formats.xls_common

logger = logging.getLogger(__name__)
default_float_factory = decimal.Decimal

# Font Size : 8pt * 20 = 160
# font = 'font: name Arial Narrow, height 160'
font = 'font: name Verdana, height 160'

if xlwt is not None:
    sty_header = xlwt.easyxf(font + ', bold on; align: vertical center, horizontal center',
                             'pattern: pattern solid, fore-colour rose')
    sty_norm = xlwt.easyxf(font + ', colour black')
    sty_first_frame = xlwt.easyxf(font + ', colour black; borders: top thin')
    sty_white = xlwt.easyxf(font + ', colour white')

    # ECU Matrix-Styles
    sty_green = xlwt.easyxf('pattern: pattern solid, fore-colour light_green')
    sty_green_first_frame = xlwt.easyxf('pattern: pattern solid, fore-colour light_green; borders: top thin')
    sty_sender = xlwt.easyxf('pattern: pattern 0x04, fore-colour gray25')
    sty_sender_first_frame = xlwt.easyxf('pattern: pattern 0x04, fore-colour gray25; borders: top thin')
    sty_sender_green = xlwt.easyxf('pattern: pattern 0x04, fore-colour gray25, back-colour light_green')
    sty_sender_green_first_frame = xlwt.easyxf(
        'pattern: pattern 0x04, fore-colour gray25, back-colour light_green; borders: top thin')


def write_ecu_matrix(ecus, sig, frame, worksheet, row, col, first_frame):

    print("def : format - xls - write_ecu_matrix")

    # type: (typing.Sequence[str], typing.Optional[canmatrix.Signal], canmatrix.Frame, xlwt.Worksheet, int, int, xlwt.XFStyle) -> int
    # first-frame - style with borders:
    if first_frame == sty_first_frame:
        norm = sty_first_frame
        sender = sty_sender_first_frame
        norm_green = sty_green_first_frame
        sender_green = sty_sender_green_first_frame
    # consecutive-frame - style without borders:
    else:
        norm = sty_norm
        sender = sty_sender
        norm_green = sty_green
        sender_green = sty_sender_green

    # iterate over ECUs:
    for ecu_name in ecus:
        # every second ECU with other style
        if col % 2 == 0:
            loc_style = norm
            loc_style_sender = sender
        # every second ECU with other style
        else:
            loc_style = norm_green
            loc_style_sender = sender_green

        # write "s" "r" "r/s" if signal is sent, received or send and received by ECU
        if sig and ecu_name in sig.receivers and ecu_name in frame.transmitters:
            worksheet.write(row, col, label="r/s", style=loc_style_sender)
        elif sig and ecu_name in sig.receivers:
            worksheet.write(row, col, label="r", style=loc_style)
        elif ecu_name in frame.transmitters:
            worksheet.write(row, col, label="s", style=loc_style_sender)
        else:
            worksheet.write(row, col, label="", style=loc_style)
        col += 1
    # loop over ECUs ends here
    return col


def write_excel_line(worksheet, row, col, row_array, style):

    print("def : format - xls - write_excel_line")

    # type: (xlwt.Worksheet, int, int, typing.Sequence, xlwt.XFStyle) -> int
    for item in row_array:
        worksheet.write(row, col, label=item, style=style)
        col += 1
    return col

def dump(db, file, **options):

    print("def : format - xls - dump")

    # type: (canmatrix.CanMatrix, typing.IO, **typing.Any) -> None
    head_top = ['ID', 'Frame Name', 'Cycle Time [ms]', 'Launch Type', 'Launch Parameter', 'Signal Byte No.',
                'Signal Bit No.', 'Signal Name', 'Signal Function', 'Signal Length [Bit]', 'Signal Default',
                ' Signal Not Available', 'Byteorder']
    head_tail = ['Value', 'Name / Phys. Range', 'Function / Increment Unit']

    if len(options.get("additionalSignalAttributes", "")) > 0:
        additional_signal_columns = options.get("additionalSignalAttributes").split(",")  # type: typing.List[str]
    else:
        additional_signal_columns = []  # ["attributes['DisplayDecimalPlaces']"]

    if len(options.get("additionalFrameAttributes", "")) > 0:
        additional_frame_columns = options.get("additionalFrameAttributes").split(",")  # type: typing.List[str]
    else:
        additional_frame_columns = []  # ["attributes['DisplayDecimalPlaces']"]

    motorola_bit_format = options.get("xlsMotorolaBitFormat", "msbreverse")

    workbook = xlwt.Workbook(encoding='utf8')
    #    ws_name = os.path.basename(filename).replace('.xls', '')
    #    worksheet = workbook.add_sheet('K-Matrix ' + ws_name[0:22])
    worksheet = workbook.add_sheet('K-Matrix ')

    row_array = []  # type: typing.List[str]
    col = 0

    # write ECUs in first row:
    ecu_list = [ecu.name for ecu in db.ecus]

    row_array += head_top
    head_start = len(row_array)

    row_array += ecu_list
    for col in range(len(row_array)):
        worksheet.col(col).width = 1111
    tail_start = len(row_array)
    row_array += head_tail

    additional_frame_start = len(row_array)

    for col in range(tail_start, len(row_array)):
        worksheet.col(col).width = 3333

    for additionalCol in additional_frame_columns:
        row_array.append("frame." + additionalCol)
        col += 1

    for additionalCol in additional_signal_columns:
        row_array.append("signal." + additionalCol)
        col += 1

    write_excel_line(worksheet, 0, 0, row_array, sty_header)

    # set width of selected Cols
    worksheet.col(1).width = 5555
    worksheet.col(3).width = 3333
    worksheet.col(7).width = 5555
    worksheet.col(8).width = 7777
    worksheet.col(head_start).width = 1111
    worksheet.col(head_start + 1).width = 5555

    frame_hash = {}
    if db.type == canmatrix.matrix_class.CAN:
        logger.debug("Length of db.frames is %d", len(db.frames))
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                logger.error("export complex multiplexers is not supported - ignoring frame %s", frame.name)
                continue
            frame_hash[int(frame.arbitration_id.id)] = frame
    else:
        frame_hash = {a.name: a for a in db.frames}

    # set row to first Frame (row = 0 is header)
    row = 1

    # iterate over the frames
    for idx in sorted(frame_hash.keys()):

        frame = frame_hash[idx]
        frame_style = sty_first_frame

        # sort signals:
        sig_hash = {"{:02d}{}".format(sig.get_startbit(), sig.name): sig for sig in frame.signals}

        # set style for first line with border
        sig_style = sty_first_frame

        additional_frame_info = [frame.attribute(frameInfo, default="") for frameInfo in additional_frame_columns]

        # iterate over signals
        row_array = []
        if len(sig_hash) == 0:  # Frames without signals
            row_array += canmatrix.formats.xls_common.get_frame_info(db, frame)
            for _ in range(5, head_start):
                row_array.append("")
            temp_col = write_excel_line(worksheet, row, 0, row_array, frame_style)
            temp_col = write_ecu_matrix(ecu_list, None, frame, worksheet, row, temp_col, frame_style)

            row_array = []
            for col in range(temp_col, additional_frame_start):
                row_array.append("")
            row_array += additional_frame_info
            for _ in additional_signal_columns:
                row_array.append("")
            write_excel_line(worksheet, row, temp_col, row_array, frame_style)
            row += 1
            continue

        # iterate over signals
        for sig_idx in sorted(sig_hash.keys()):
            sig = sig_hash[sig_idx]

            # if not first Signal in Frame, set style
            if sig_style != sty_first_frame:
                sig_style = sty_norm

            if sig.values.__len__() > 0:  # signals with value table
                val_style = sig_style
                # iterate over values in value table
                for val in sorted(sig.values.keys()):
                    row_array = canmatrix.formats.xls_common.get_frame_info(db, frame)
                    front_col = write_excel_line(worksheet, row, 0, row_array, frame_style)
                    if frame_style != sty_first_frame:
                        worksheet.row(row).level = 1

                    col = head_start
                    col = write_ecu_matrix(ecu_list, sig, frame, worksheet, row, col, frame_style)

                    # write Value
                    (frontRow, backRow) = canmatrix.formats.xls_common.get_signal(db, frame, sig, motorola_bit_format)
                    write_excel_line(worksheet, row, front_col, frontRow, sig_style)
                    backRow += additional_frame_info
                    for item in additional_signal_columns:
                        temp = getattr(sig, item, "")
                        backRow.append(temp)

                    write_excel_line(worksheet, row, col + 2, backRow, sig_style)
                    write_excel_line(worksheet, row, col, [val, sig.values[val]], val_style)

                    # no min/max here, because min/max has same col as values...
                    # next row
                    row += 1
                    # set style to normal - without border
                    sig_style = sty_white
                    frame_style = sty_white
                    val_style = sty_norm
                # loop over values ends here
            # no value table available
            else:
                row_array = canmatrix.formats.xls_common.get_frame_info(db, frame)
                front_col = write_excel_line(worksheet, row, 0, row_array, frame_style)
                if frame_style != sty_first_frame:
                    worksheet.row(row).level = 1

                col = head_start
                col = write_ecu_matrix(
                    ecu_list, sig, frame, worksheet, row, col, frame_style)
                (frontRow, backRow) = canmatrix.formats.xls_common.get_signal(db, frame, sig, motorola_bit_format)
                write_excel_line(worksheet, row, front_col, frontRow, sig_style)

                if float(sig.min) != 0 or float(sig.max) != 1.0:
                    backRow.insert(0, str("%g..%g" % (sig.min, sig.max)))  # type: ignore
                else:
                    backRow.insert(0, "")
                backRow.insert(0, "")

                backRow += additional_frame_info
                for item in additional_signal_columns:
                    temp = getattr(sig, item, "")
                    backRow.append(temp)

                write_excel_line(worksheet, row, col, backRow, sig_style)

                # next row
                row += 1
                # set style to normal - without border
                sig_style = sty_white
                frame_style = sty_white
        # loop over signals ends here
    # loop over frames ends here

    # frozen headings instead of split panes
    worksheet.set_panes_frozen(True)
    # in general, freeze after last heading row
    worksheet.set_horz_split_pos(1)
    worksheet.set_remove_splits(True)
    # save file
    workbook.save(file)


# ########################### load ###############################

def parse_value_name_column(value_name, value_str, signal_size, float_factory):

    print("def : format - xls - parse_value_name_column")

    # type: (str, str, int, typing.Callable) -> typing.Tuple
    mini = maxi = offset = None  # type: typing.Any
    value_table = dict()
    if ".." in value_name:
        (mini, maxi) = value_name.strip().split("..")
        mini = float_factory(mini)
        maxi = float_factory(maxi)
        offset = mini

    elif len(value_name) > 0:
        if len(value_str.strip()) > 0:
            # Value Table
            value = int(float(value_str))
            value_table[value] = value_name
        maxi = pow(2, signal_size) - 1
        maxi = float_factory(maxi)
        mini = 0
        offset = 0
    return mini, maxi, offset, value_table


def read_additional_signal_attributes(signal, attribute_name, attribute_value):

    print("def : format - xls - read_additional_signal_attributes")

    if not attribute_name.startswith("signal"):
        return
    if attribute_name.replace("signal.", "") in vars(signal):
        command_str = attribute_name + "="
        command_str += str(attribute_value)
        if len(str(attribute_value)) > 0:
            exec(command_str)
    else:
        pass


def load(file, **options):

    print("def : format - xls - load")

    # type: (typing.IO, **typing.Any) -> canmatrix.CanMatrix
    wb = xlrd.open_workbook(file_contents=file.read())
    sh = wb.sheet_by_index(0)
    db = canmatrix.CanMatrix()

    # Defines not imported... add by users
    # add_ecu_defines BA_DEF_ BU_
    #db.add_ecu_defines("NWM-Stationsadresse", 'HEX 0 63')
    #db.add_ecu_defines("NWM-Knoten", 'ENUM  "nein","ja"')

    # add_frame_defines BA_DEF_ BO_
    #db.add_frame_defines("GenMsgDelayTime", 'INT 0 65535')
    #db.add_frame_defines("GenMsgCycleTimeActive", 'INT 0 65535')
    #db.add_frame_defines("GenMsgNrOfRepetitions", 'INT 0 65535')
    #db.add_frame_defines("GenMsgStartValue", 'STRING')

    # add_signal_defines BA_DEF_ SG_
    #db.add_signal_defines("GenSigSNA", 'STRING')

    SignalGroupTemp = []
    FrameTemp = []

    FrameWithSignal = []

    # eval search for correct columns:
    index = {}
    for i in range(sh.ncols):
        value = sh.cell(0, i).value
        if value == "Transmitter":
            index['Transmitter'] = i
        elif "Receiver" in value:
            index['Receiver'] = i
        elif "Signal Name" in value:
            index['SignalName'] = i
        elif "Message Name" in value:
            index['MessageName'] = i
        elif "Message ID" in value:
            index['MessageID'] = i
        elif "Message Type" in value:
            index['MessageType'] = i
        elif "Period [ms]" in value:
            index['Period'] = i
        elif "DLC [Byte]" in value:
            index['DLC'] = i
        elif "MSB" in value:
            index['MSB'] = i
        elif "LSB" in value:
            index['LSB'] = i
        elif "Size [Bit]" in value:
            index['Size'] = i
        elif "Byte Order" in value:
            print("def : format - xls - load - BYTE ORDER INDEX: {} ".format(i))
            index['ByteOrder'] = i
        elif "Datatype" in value:
            index['DataType'] = i
        elif "Default Initialized Value" in value:
            index['DefaultInitializedValue'] = i
        elif "Factor" in value:
            index['Factor'] = i
        elif "Offset" in value:
            index['Offset'] = i
        elif "P-Minimum" in value:
            index['Minimum'] = i
        elif "P-Maximum" in value:
            index['Maximum'] = i
        elif "Unit" == value:
            print("def : format - xls - load - UNIT INDEX: {} ".format(i))
            index['Unit'] = i
        elif "Coding" in value:
            index['Coding'] = i
        elif "Signal Group" in value:
            index['Signal Group'] = i
        elif "Comment" in value:
            index['Comment'] = i
        # elif "Multiplexer" in value:
        #     print("def : format - xls - load - MULTIPLEXER INDEX: {} ".format(i))
        #     index['Multiplexer'] = i

    index['ECUstart'] = index['Comment'] + 1
    index['ECUend'] = sh.ncols

    # ECUs:
    for x in range(index['ECUstart'], index['ECUend']):
        print("def : format - xls - load - ECU : {}".format(sh.cell(0, x).value))
        db.add_ecu(canmatrix.Ecu(sh.cell(0, x).value))

    # initialize:
    frame_id = None
    signal_name = None

    # read xls/xlsx row by row
    for row_num in range(1, sh.nrows):
        # new signal detected
        if sh.cell(row_num, index['SignalName']).value.strip() != signal_name or \
                sh.cell(row_num, index['MessageID']).value.strip() != frame_id:

            # receiver
            receiver = [sh.cell(row_num, index['Receiver']).value]

            # signal name
            signal_name = sh.cell(row_num, index['SignalName']).value.strip()

            # MSB
            msb = int(sh.cell(row_num, index['MSB']).value)

            # LSB
            lsb = int(sh.cell(row_num, index['LSB']).value)

            # size
            signal_length = int(sh.cell(row_num, index['Size']).value)

            # cycle_time
            cycle_time = 0

            # byte order
            if index.get("ByteOrder", False):
                signal_byte_order = sh.cell(row_num, index['ByteOrder']).value

                if 'intel' in signal_byte_order:
                    is_little_endian = True
                else:
                    is_little_endian = False
            else:
                is_little_endian = True  # Default Intel

            # datatype
            is_signed = sh.cell(row_num, index['DataType']).value

            # default initialized value (not use in dbc file)
            signal_default = sh.cell(row_num, index['DefaultInitializedValue']).value

            # factor
            factor = str(sh.cell(row_num, index['Factor']).value)
            factor = decimal.Decimal(factor)

            # offset
            offset = str(sh.cell(row_num, index['Offset']).value)
            offset = decimal.Decimal(offset)

            # p-minimum
            pminimum = str(sh.cell(row_num, index['Minimum']).value)
            pminimum = decimal.Decimal(pminimum)

            # p-maximum
            pmaximum = str(sh.cell(row_num, index['Maximum']).value)
            pmaximum = decimal.Decimal(pmaximum)

            # unit
            unit = sh.cell(row_num, index['Unit']).value
            print("def : format - xls - load - unit : {} - {} - {}".format(row_num,unit,index['Unit']))

            # coding
            coding = sh.cell(row_num, index['Coding']).value
            # eval coding (value table)
            value_table = dict()
            coding = coding.replace('\n', ':').split(':')

            for i in range(len(coding)):
                if coding[i].isdigit():
                    value_table[coding[i]] = coding[i + 1]
                else:
                    pass


            # # message name
            # frame_name = sh.cell(row_num, index['MessageName']).value
            #
            # signal_group = sh.cell(row_num, index['Signal Group']).value
            # signal_group_find = False
            # # Frame Name = 0 , Signal Group Name = 1 , Signal Group ID = 2 , Signals = 3
            # if signal_group != '/':
            #     for each in SignalGroupTemp:
            #         if each[1] == signal_group:
            #             signal_group_find = True
            #             if len(each[3]) == 0:
            #                 each[3] = []
            #                 each[3].append(signal_name)
            #             else:
            #                 each[3].append(signal_name)
            #     if signal_group_find == False:
            #         SignalGroupTemp.append([frame_name,signal_group,1,[signal_name]])

            # comment
            signal_comment = sh.cell(row_num, index['Comment']).value.strip()

            # signal_multiplexer = sh.cell(row_num, index['Multiplexer']).value

            signal_multiplexer = '/'
            if signal_multiplexer == '/':

                # create Canmatrix signal and add to db
                new_signal = canmatrix.Signal(
                    name=signal_name,
                    msb=msb,
                    lsb=lsb,
                    size=int(signal_length),
                    cycle_time=cycle_time,
                    is_little_endian=is_little_endian,
                    is_signed=is_signed,
                    offset=offset,
                    factor=factor,
                    receivers=receiver,
                    unit=unit,
                    min=pminimum,
                    max=pmaximum,
                    values=value_table)

                new_signal.add_comment(signal_comment)

                print("def : format - xls - load - SIGNAL - {} - {}".format(new_signal.name,new_signal.values))
                db.add_signal(new_signal)

            else:

                if signal_multiplexer != "Multiplexor":
                    # create Canmatrix signal and add to db
                    new_signal = canmatrix.Signal(
                        name=signal_name,
                        msb=msb,
                        lsb=lsb,
                        size=int(signal_length),
                        cycle_time=cycle_time,
                        is_little_endian=is_little_endian,
                        is_signed=is_signed,
                        offset=offset,
                        factor=factor,
                        receivers=receiver,
                        unit=unit,
                        min=pminimum,
                        max=pmaximum,
                        multiplex=int(signal_multiplexer),
                        values=value_table)

                    new_signal.is_multiplexer = True
                    new_signal.add_comment(signal_comment)

                    print("def : format - xls - load - SIGNAL - {} - {}".format(new_signal.name, new_signal.values))
                    db.add_signal(new_signal)
                else:
                    # create Canmatrix signal and add to db
                    new_signal = canmatrix.Signal(
                        name=signal_name,
                        msb=msb,
                        lsb=lsb,
                        size=int(signal_length),
                        cycle_time=cycle_time,
                        is_little_endian=is_little_endian,
                        is_signed=is_signed,
                        offset=offset,
                        factor=factor,
                        receivers=receiver,
                        unit=unit,
                        min=pminimum,
                        max=pmaximum,
                        values=value_table)

                    new_signal.is_multiplexer = True
                    new_signal.multiplex = 'Multiplexor'
                    new_signal.add_comment(signal_comment)

                    print("def : format - xls - load - SIGNAL - {} - {}".format(new_signal.name, new_signal.values))
                    db.add_signal(new_signal)

            # Frame information for new signal

            # message name
            frame_name = sh.cell(row_num, index['MessageName']).value

            signal_group = sh.cell(row_num, index['Signal Group']).value
            signal_group_find = False
            # Frame Name = 0 , Signal Group Name = 1 , Signal Group ID = 2 , Signals = 3
            if signal_group != '/':
                for each in SignalGroupTemp:
                    if each[1] == signal_group:
                        signal_group_find = True
                        if len(each[3]) == 0:
                            each[3] = []
                            each[3].append(new_signal)
                        else:
                            each[3].append(new_signal)
                if signal_group_find == False:
                    SignalGroupTemp.append([frame_name,signal_group,1,[new_signal]])

            # message id
            frame_id = sh.cell(row_num, index['MessageID']).value

            # period (follow signal period)
            cycle_time = int(sh.cell(row_num, index['Period']).value)
            # dlc
            dlc = int(sh.cell(row_num, index['DLC']).value)

            # transmitter
            transmitter = [sh.cell(row_num, index['Transmitter']).value]

            # create Canmatrix frame
            new_frame: object = canmatrix.Frame(frame_name, size=dlc)
            
            new_frame.cycle_time = cycle_time

            # eval transmitter
            new_frame.transmitters = transmitter

            # eval message ID
            if frame_id.endswith("xh"):
                new_frame.arbitration_id = canmatrix.ArbitrationId(int(frame_id[:-2], 16), extended=True)
            else:
                new_frame.arbitration_id = canmatrix.ArbitrationId(int(frame_id[2:], 16), extended=False)

            if sh.cell(row_num, index['MessageType']).value == "EXT":
                #frame.arbitration_id.id = (int(temp.group(1)) - 0x80000000)
                new_frame.arbitration_id.extended = True

            # add frame information to signal
            new_signal.frames = new_frame
            FrameTemp.append(new_frame)

            if signal_name != '/':
                FrameWithSignal.append(new_frame)

    # read rows values loop end

    # add vframeformat
    for signal in db.signals:
        signal.frames.set_fd_type()

    SkipUsedFrame = []
    for each in  SignalGroupTemp:
        #print("def : format - xls - load - SIGNAL GROUP - START - {}".format(each[0]))
        #print("def : format - xls - load - SIGNAL GROUP - START - ",db.signals)
        for each_frame in FrameTemp:
            #print("def : format - xls - load - SIGNAL GROUP - SCAN FRAME - {} - {}".format(each[0],each_frame.name))
            if each_frame.name == each[0] and each_frame.name not in SkipUsedFrame:
                SkipUsedFrame.append(each_frame.name)
                print("def : format - xls - load - SIGNAL GROUP - SCAN FRAME - {} - {}".format(each[1], each[0]))
                each_frame.add_signal_group(each[1],1,each[3])
                print("def : format - xls - load - SIGNAL GROUP - ", each[3])

            #print("def : format - xls - load - SIGNAL GROUP - {}".format(each_frame))
            #each_frame.add_signal_group("A",1,["Hello"])


    #WRITE DEF
    # sh_def = wb.sheet_by_index(1)
    #
    # # eval search for correct columns:
    # index = {}
    # for i in range(sh_def.ncols):
    #     value = sh_def.cell(0, i).value
    #     if value == "DEF TYPE":
    #         index['DEFTYPE'] = i
    #     elif "VALUE" in value:
    #         index['VALUE'] = i
    #     elif "TYPE" in value:
    #         index['TYPE'] = i
    #     elif "MIN" in value:
    #         index['MIN'] = i
    #     elif "MAX" in value:
    #         index['MAX'] = i
    #     elif "DEFAULT" in value:
    #         index['DEFAULT'] = i
    #
    # for row_num in range(1, sh_def.nrows):
    #
    #     #print("DEF ROW : {} - {} - {}".format(sh_def.cell(row_num, index['DEFTYPE']).value,sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['TYPE']).value))
    #
    #     if sh_def.cell(row_num, index['DEFTYPE']).value == "GLO DEF":
    #         #print("DEF ROW - G : {} {} ".format(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['TYPE']).value))
    #
    #         if sh_def.cell(row_num, index['TYPE']).value == "HEX":
    #
    #             #print("DEF ROW - G : {} {} {}".format(sh_def.cell(row_num, index['VALUE']).value,int(sh_def.cell(row_num, index['MIN']).value),int(sh_def.cell(row_num, index['MAX']).value)))
    #             db.add_global_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,str(int(sh_def.cell(row_num, index['MIN']).value)),str(int(sh_def.cell(row_num, index['MAX']).value))))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "INT":
    #             #print("DEF ROW - G - INT : {} - {}".format(sh_def.cell(row_num, index['VALUE']).value,int(sh_def.cell(row_num, index['DEFAULT']).value)))
    #             db.add_global_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,str(int(sh_def.cell(row_num, index['MIN']).value)),str(int(sh_def.cell(row_num, index['MAX']).value))))
    #         elif sh_def.cell(row_num, index['TYPE']).value == "STRING":
    #             print("DEF ROW - G - STRING : {} - {}".format(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['DEFAULT']).value))
    #             db.add_global_defines(sh_def.cell(row_num, index['VALUE']).value, sh_def.cell(row_num, index['TYPE']).value)
    #
    #         if sh_def.cell(row_num, index['DEFAULT']).value != '/':
    #             print("DEF ROW - TRACE : {} - {} ".format(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['DEFAULT']).value))
    #             #db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['DEFAULT']).value)
    #         else:
    #             db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,"")
    #
    #
    #     elif sh_def.cell(row_num, index['DEFTYPE']).value == "ECU DEF":
    #
    #         if sh_def.cell(row_num, index['TYPE']).value == "INT":
    #             db.add_ecu_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,int(sh_def.cell(row_num, index['MIN']).value),int(sh_def.cell(row_num, index['MAX']).value)))
    #             # if sh_def.cell(row_num, index['DEFAULT']).value != '/':
    #             #     db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,"0")
    #             # else:
    #             #     db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,"")
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "HEX":
    #
    #             db.add_ecu_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,int(sh_def.cell(row_num, index['MIN']).value),int(sh_def.cell(row_num, index['MAX']).value)))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "ENUM":
    #
    #             db.add_ecu_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {}".format(sh_def.cell(row_num, index['TYPE']).value,sh_def.cell(row_num, index['MIN']).value))
    #
    #         else:
    #             db.add_ecu_defines(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['TYPE']).value)
    #
    #
    #         if sh_def.cell(row_num, index['DEFAULT']).value != '/':
    #
    #             #print("DEBUG : {}".format(sh_def.cell(row_num, index['DEFAULT']).value))
    #             db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,str(int(sh_def.cell(row_num, index['DEFAULT']).value)))
    #         else:
    #             db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,"")
    #
    #     elif sh_def.cell(row_num, index['DEFTYPE']).value == "ENV DEF":
    #
    #         if sh_def.cell(row_num, index['TYPE']).value == "ENUM":
    #
    #             db.add_env_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {}".format(sh_def.cell(row_num, index['TYPE']).value,sh_def.cell(row_num, index['MIN']).value))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "INT":
    #
    #             db.add_env_defines(sh_def.cell(row_num, index['VALUE']).value,
    #                                "{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,
    #                                               int(sh_def.cell(row_num, index['MIN']).value),
    #                                                  int(sh_def.cell(row_num, index['MAX']).value)))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "STRING":
    #
    #             db.add_env_defines(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['TYPE']).value)
    #
    #     elif sh_def.cell(row_num, index['DEFTYPE']).value == "FRM DEF":
    #
    #         if sh_def.cell(row_num, index['TYPE']).value == "INT":
    #
    #             db.add_frame_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,int(sh_def.cell(row_num, index['MIN']).value),int(sh_def.cell(row_num, index['MAX']).value)))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "HEX":
    #             pass
    #         elif sh_def.cell(row_num, index['TYPE']).value == "STRING":
    #
    #             db.add_frame_defines(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['TYPE']).value)
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "FLOAT":
    #             db.add_frame_defines(sh_def.cell(row_num, index['VALUE']).value,
    #                                  "{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,
    #                                                    sh_def.cell(row_num, index['MIN']).value,
    #                                                    sh_def.cell(row_num, index['MAX']).value))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "ENUM":
    #
    #             db.add_frame_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {}".format(sh_def.cell(row_num, index['TYPE']).value,sh_def.cell(row_num, index['MIN']).value))
    #
    #     elif sh_def.cell(row_num, index['DEFTYPE']).value == "SIG DEF":
    #
    #         if sh_def.cell(row_num, index['TYPE']).value == "INT":
    #
    #             db.add_signal_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,int(sh_def.cell(row_num, index['MIN']).value),int(sh_def.cell(row_num, index['MAX']).value)))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "HEX":
    #             pass
    #         elif sh_def.cell(row_num, index['TYPE']).value == "STRING":
    #
    #             db.add_signal_defines(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['TYPE']).value)
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "FLOAT":
    #             db.add_signal_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {} {}".format(sh_def.cell(row_num, index['TYPE']).value,int(sh_def.cell(row_num, index['MIN']).value),int(sh_def.cell(row_num, index['MAX']).value)))
    #
    #         elif sh_def.cell(row_num, index['TYPE']).value == "ENUM":
    #             #print("DEBUG : SIG STRING ENUM : {}".format(sh_def.cell(row_num, index['MIN']).value))
    #             db.add_signal_defines(sh_def.cell(row_num, index['VALUE']).value,"{} {}".format(sh_def.cell(row_num, index['TYPE']).value,sh_def.cell(row_num, index['MIN']).value))
    #
    # #WRITE BA_DEF_DEF_
    #
    # for row_num in range(1, sh_def.nrows):
    #     if sh_def.cell(row_num, index['DEFTYPE']).value == "BA_DEF_DEF_":
    #         print("def : format - xls - load - BA_DEF_DEF_ : {} - {} ".format(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['DEFAULT']).value))
    #         if sh_def.cell(row_num, index['DEFAULT']).value == "/":
    #             #print("def : format - dbc - load - BA_DEF_DEF_ : {} - {} ".format(sh_def.cell(row_num, index['VALUE']).value,''))
    #             db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,"")
    #         else:
    #             db.add_define_default(sh_def.cell(row_num, index['VALUE']).value,sh_def.cell(row_num, index['DEFAULT']).value)
    #
    # #WRITE ATTR
    # sh_attr = wb.sheet_by_index(2)
    #
    # # eval search for correct columns:
    # index = {}
    # for i in range(sh_attr.ncols):
    #     value = sh_attr.cell(0, i).value
    #     if value == "P1":
    #         index['P1'] = i
    #     elif "P2" in value:
    #         index['P2'] = i
    #     elif "P3" in value:
    #         index['P3'] = i
    #     elif "P4" in value:
    #         index['P4'] = i
    #     elif "P5" in value:
    #         index['P5'] = i
    #
    # for row_num in range(1, sh_attr.nrows):
    #
    #     if sh_attr.cell(row_num, index['P2']).value == "BU_":
    #         db.ecu_by_name(sh_attr.cell(row_num, index['P3']).value).add_attribute(
    #             sh_attr.cell(row_num, index['P1']).value,
    #             sh_attr.cell(row_num, index['P4']).value)
    #
    #     elif sh_attr.cell(row_num, index['P2']).value == "BO_":
    #
    #         #for each in FrameTemp:
    #         #for each in FrameWithSignal:
    #         for each in FrameTemp:
    #             # print("DEBUG : ADD FRAME ATTR - {} - {}".format(each.arbitration_id.id,
    #             #
    #             #
    #             #                                                sh_attr.cell(row_num, index['P3']).value))
    #             if each.arbitration_id.extended == False:
    #                 if str(each.arbitration_id.id) == sh_attr.cell(row_num, index['P3']).value:
    #                     # if sh_attr.cell(row_num, index['P1']).value != "GenMsgILSupport" or sh_attr.cell(row_num, index['P1']).value != "Diag_Response":
    #
    #                     print("DEBUG : ADD STD FRAME ATTR - {} - {} - {}".format(each.name,sh_attr.cell(row_num, index['P1']).value,sh_attr.cell(row_num, index['P4']).value))
    #                     if sh_attr.cell(row_num, index['P4']).value == "/":
    #                         each.add_attribute(sh_attr.cell(row_num, index['P1']).value,"")
    #                     else:
    #                         each.add_attribute(sh_attr.cell(row_num, index['P1']).value,sh_attr.cell(row_num, index['P4']).value)
    #             else:
    #                 if str(each.arbitration_id.id + 0x80000000) == sh_attr.cell(row_num, index['P3']).value:
    #                     # if sh_attr.cell(row_num, index['P1']).value != "GenMsgILSupport" or sh_attr.cell(row_num, index['P1']).value != "Diag_Response":
    #
    #                     print("DEBUG : ADD EXT FRAME ATTR - {} - {} - {}".format(each.name,
    #                                                                          sh_attr.cell(row_num, index['P1']).value,
    #                                                                          sh_attr.cell(row_num, index['P4']).value))
    #                     if sh_attr.cell(row_num, index['P4']).value == "/":
    #                         each.add_attribute(sh_attr.cell(row_num, index['P1']).value, "")
    #                     else:
    #                         each.add_attribute(sh_attr.cell(row_num, index['P1']).value,
    #                                            sh_attr.cell(row_num, index['P4']).value)
    #
    #         # get_frame_by_id(canmatrix.ArbitrationId.from_compound_integer(sh_attr.cell(row_num, index['P3']).value)).add_attribute(
    #         #     sh_attr.cell(row_num, index['P1']).value,
    #         #     sh_attr.cell(row_num, index['P4']).value)
    #
    #     elif sh_attr.cell(row_num, index['P2']).value == "SG_":
    #         FrameIdandSignal = (sh_attr.cell(row_num, index['P3']).value).split(" ")
    #         print("DEBUG : ADD SIGNAL ATTR - BA_ SG_ - {} - {}".format(FrameIdandSignal[0],FrameIdandSignal[1]))
    #         for signal in db.signals:
    #             if signal.name == FrameIdandSignal[1] and signal.frames.arbitration_id.id == int(FrameIdandSignal[0]):
    #                 print("DEBUG : ADD SIGNAL ATTR - BA_ SG_ - {} ; {}".format(sh_attr.cell(row_num, index['P1']).value,sh_attr.cell(row_num, index['P4']).value))
    #                 signal.add_attribute(sh_attr.cell(row_num, index['P1']).value,sh_attr.cell(row_num, index['P4']).value)
    #
    #     elif sh_attr.cell(row_num, index['P2']).value == "GLO":
    #         db.add_attribute(sh_attr.cell(row_num, index['P1']).value, sh_attr.cell(row_num, index['P4']).value)
    #
    #     elif sh_attr.cell(row_num, index['P2']).value == "SIG_VALTYPE_":
    #         for signal in db.signals:
    #             if signal.name == sh_attr.cell(row_num, index['P4']).value and signal.frames.arbitration_id.id == int(sh_attr.cell(row_num, index['P3']).value):
    #                 signal.is_float = True

    # ecu-attributes:
    # for ecu in db.ecus:
    #     for attrib, val in sorted(ecu.attributes.items()):
    #
    #         print("def : format - dbc - dump - ECU ATTRIBUTE : [{} , {}]".format(attrib, val))

            # f.write(
            #     create_attribute_string(attrib, "BU_", ecu.name, val, db.ecu_defines[attrib].type == "STRING").encode(
            #         dbc_export_encoding, ignore_encoding_errors))

    # # global-attributes:
    # for attrib, val in sorted(db.attributes.items()):
    #
    #     print("def : format - dbc - dump - GLOBAL ATTRIBUTE : [{} , {}]".format(attrib, val))
    #
    #     f.write(create_attribute_string(attrib, "", "", val, db.global_defines[attrib].type == "STRING").encode(
    #         dbc_export_encoding, ignore_encoding_errors))
    #
    # # messages-attributes:
    # curr_frame = None
    # for signal in db.signals:
    #     frame = signal.frames
    #     if frame.name != curr_frame:
    #         for attrib, val in sorted(frame.attributes.items()):
    #
    #             print("def : format - dbc - dump - MESSAGE ATTRIBUTE : [{} , {}]".format(attrib,val))
    #
    #             f.write(create_attribute_string(attrib, "BO_", str(frame.arbitration_id.to_compound_integer()), val,
    #                                             db.frame_defines[attrib].type == "STRING").encode(dbc_export_encoding,
    #                                                                                               ignore_encoding_errors))
    #         curr_frame = frame.name
    #
    # # signal-attributes:
    # for signal in db.signals:
    #     frame = signal.frames
    #     for attrib, val in sorted(signal.attributes.items()):
    #         name = output_names[frame][signal]
    #         if isinstance(val, float):
    #             val = format_float(val)
    #         if attrib in db.signal_defines:
    #
    #             print("def : format - dbc - dump - SIGNAL ATTRIBUTE : {}".format(attrib))
    #
    #             f.write(create_attribute_string(
    #                 attrib, "SG_", '%d ' % frame.arbitration_id.to_compound_integer() + name, val,
    #                                db.signal_defines[attrib].type == "STRING").encode(dbc_export_encoding,
    #                                                                                   ignore_encoding_errors))
    #
    # # environment-attributes:
    # for env_var_name, env_var in db.env_vars.items():
    #     if "attributes" in env_var:
    #         for attribute, value in env_var["attributes"].items():
    #
    #             print("def : format - dbc - dump - ENVIRONMENT ATTRIBUTE : [{} , {}]".format(attrib, val))
    #
    #             f.write(create_attribute_string(attribute, "EV_", "", value,
    #                                             db.env_defines[attribute].type == "STRING")
    #                     .encode(dbc_export_encoding, ignore_encoding_errors))


    return db
