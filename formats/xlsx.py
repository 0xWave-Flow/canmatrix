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

import logging
import typing
from builtins import *

import xlsxwriter

import canmatrix
import canmatrix.formats.xls_common

logger = logging.getLogger(__name__)

# Font Size : 8pt * 20 = 160
# font = 'font: name Arial Narrow, height 160'
font = 'font: name Verdana, height 160'

sty_header = 0
sty_norm = 0
sty_first_frame = 0
sty_white = 0

sty_green = 0
sty_green_first_frame = 0
sty_sender = 0
sty_sender_first_frame = 0
sty_sender_green = 0
sty_sender_green_first_frame = 0


def write_excel_line(worksheet, row, col, row_array, style):

    print("def : format - xlsx - write_excel_line - {}".format(row_array))

    # type: (xlsxwriter.workbook.Worksheet, int, int, typing.Sequence[typing.Any], xlsxwriter.workbook.Format) -> int
    for item in row_array:
        worksheet.write(row, col, item, style)
        col += 1
    return col


def dump(db, filename, **options):

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>def : format - xlsx - dump")

    # type: (canmatrix.CanMatrix, str, **str) -> None
    motorola_bit_format = options.get("xlsMotorolaBitFormat", "msbreverse")

    # add excel head fields
    head_top = [
        'Result',
        'No.',
        'Transmitter',
        'Receiver',
        'Signal Name',
        'ASIL Level',
        'Signal Description',
        'Used by function',
        'Message Name',
        'Message ID',
        'Message Type',
        'Signal Type',
        'Detailed description \nof E/M Message sending \nbehavior Tx',
        'Detailed description \nof E/M Message sending \nbehavior Rx',
        'Period [ms]',
        'Signal latency budget \nTx [ms]',
        'Signal latency budget \nRx [ms]',
        'DLC [Byte]',
        'MSB',
        'LSB',
        'Size [Bit] ',
        'Byte Order',
        'Datatype',
        'First valid signal \nduration Tx[ms]',
        'First valid signal \nduration Rx[ms]',
        'Default Initialized Value',
        'Alternative value',
        'Factor',
        'Measured Resolution Tx',
        'Measured Resolution Rx',
        'Accuracy Tx',
        'Offset',
        'P-Minimum',
        'P-Maximum',
        'Unit',
        'Coding',
        'Signal Group',
        'Message Segment',
        'Variant and options',
        'Comment',
        'Multiplexer'
    ]

    workbook = xlsxwriter.Workbook(filename)
    # worksheet name
    worksheet = workbook.add_worksheet('K-Matrix ')

    # data formats
    global sty_header
    sty_header = workbook.add_format({'bold': True,
                                      'rotation': 90,
                                      'font_name': 'Verdana',
                                      'font_size': 8,
                                      'align': 'center',
                                      'valign': 'vcenter'})
    sty_header.set_text_wrap()

    global sty_first_frame
    sty_first_frame = workbook.add_format({'font_name': 'Verdana',
                                           'font_size': 8,
                                           'font_color': 'black', 'top': 1,
                                           'align': 'center',
                                           'valign': 'vcenter'
                                           })

    global sty_white
    sty_white = workbook.add_format({'font_name': 'Verdana',
                                     'font_size': 8,
                                     'font_color': 'white',
                                     'align': 'center',
                                     'valign': 'center'
                                     })

    global sty_norm
    sty_norm = workbook.add_format({'font_name': 'Verdana',
                                    'font_size': 8,
                                    'font_color': 'black',
                                    'align': 'center',
                                    'valign': 'vcenter',
                                    })
    sty_norm.set_text_wrap()

    # ECUMatrix-Styles
    global sty_green
    sty_green = workbook.add_format({'pattern': 1, 'fg_color': '#CCFFCC'})
    global sty_green_first_frame
    sty_green_first_frame = workbook.add_format(
        {'pattern': 1, 'fg_color': '#CCFFCC', 'top': 1})
    global sty_sender
    sty_sender = workbook.add_format({'pattern': 0x04, 'fg_color': '#C0C0C0'})
    global sty_sender_first_frame
    sty_sender_first_frame = workbook.add_format(
        {'pattern': 0x04, 'fg_color': '#C0C0C0', 'top': 1})
    global sty_sender_green
    sty_sender_green = workbook.add_format(
        {'pattern': 0x04, 'fg_color': '#C0C0C0', 'bg_color': '#CCFFCC'})
    global sty_sender_green_first_frame
    sty_sender_green_first_frame = workbook.add_format(
        {'pattern': 0x04, 'fg_color': '#C0C0C0', 'bg_color': '#CCFFCC', 'top': 1})

    row_array = head_top

    # write ECUs in first row:
    ecu_list = [ecu.name for ecu in db.ecus]
    row_array += ecu_list
    for col in range(0, len(row_array)):
        worksheet.set_column(col, col, 10)

    # set width of selected Cols
    worksheet.set_column(2, 2, 12.29)
    worksheet.set_column(3, 3, 12.29)
    worksheet.set_column(19, 19, 20)

    # write head_top
    write_excel_line(worksheet, 0, 0, row_array, sty_header)

    # set row to first Frame (row = 0 is header)
    row = 1

    # iterate over the frames
    curr_frame = ''
    FrameOnly = []
    for signal in db.signals:
        print("def : format - xlsx - dump - LOOP OF SIGNAL : {}".format(signal))
        if signal.frames[0] != curr_frame:
            signal_style = sty_first_frame
            curr_frame = signal.frames[0]

        frame = signal.frames[0]

        #if (frame.arbitration_id.extended == False) or (signal.is_multiplexed == False):
        if frame.arbitration_id.extended == False:

            # get data from xls_common.py

            print("def : format - xlsx - dump - TOTAL SIGNAL DUMP : {}".format(signal))

            frontRow = canmatrix.formats.xls_common.get_signal(db, frame, signal, motorola_bit_format)

            print("def : format - xlsx - dump - FRAME STRUCT : {}".format(frontRow[9]))

            #if not (int(frontRow[9],16) >= 0x80000000):

            # write excel lines and return col
            print("def : format - xlsx - dump - ROW : {}".format(frontRow))
            col = write_excel_line(worksheet, row, 0, frontRow, signal_style)

            # add ecu receivers
            for receiver in signal.receivers:
                for [idx, name] in enumerate(ecu_list):
                    if name == receiver:
                        write_excel_line(worksheet, row, col + idx, "X", signal_style)
                    else:
                        write_excel_line(worksheet, row, col + idx, " ", signal_style)
            row += 1
            signal_style = sty_norm
            # loop over values ends here

    head_top = [
        'DEF TYPE',
        'VALUE',
        'TYPE',
        'MIN',
        'MAX',
        'DEFAULT'
    ]
    worksheet_def = workbook.add_worksheet('DEF')

    write_excel_line(worksheet_def, 0, 0, head_top, sty_header)

    # WRITE DEFINE
    row = 1
    for ENV_Define in db.env_defines:
        #print("def : format - xls_common - get_signal - ENV DEF : ",type(ENV_Define),type((db.env_defines.get(ENV_Define).type)))
        frontRow = ["ENV DEF",ENV_Define,db.env_defines.get(ENV_Define).type,'/','/','/']
        col = write_excel_line(worksheet_def, row, 0, frontRow, signal_style)
        row += 1

    for GLO_Define in db.global_defines:

        if db.global_defines.get(GLO_Define).type != "HEX":
            #print("def : format - xls_common - get_signal - GLO DEF : {} - {} ".format(GLO_Define,db.global_defines.get(GLO_Define).type))
            frontRow = ["GLO DEF",GLO_Define,db.global_defines.get(GLO_Define).type,'/','/',db.global_defines.get(GLO_Define).defaultValue]
            if db.global_defines.get(GLO_Define).defaultValue == '':
                frontRow[5] = '/'
        else:
            frontRow = ["GLO DEF", GLO_Define, db.global_defines.get(GLO_Define).type,db.global_defines.get(GLO_Define).min,db.global_defines.get(GLO_Define).max,db.global_defines.get(GLO_Define).defaultValue]
            if db.global_defines.get(GLO_Define).defaultValue == '':
                frontRow[5] = '/'

        col = write_excel_line(worksheet_def, row, 0, frontRow, signal_style)
        row += 1

    for ECU_Define in db.ecu_defines:
        print("def : format - xls_common - get_signal - ECU DEF : {}".format(db.ecu_defines.get(ECU_Define).defaultValue))
        #print("def : format - xls_common - get_signal - DEFAULT : ",db.signal_defines)

        frontRow = ["ECU DEF",ECU_Define,db.ecu_defines.get(ECU_Define).type,'/','/','/']

        if db.ecu_defines.get(ECU_Define).type != "ENUM":
            if db.ecu_defines.get(ECU_Define).min == None:
                frontRow[3] = '/'
            else:
                frontRow[3] = db.ecu_defines.get(ECU_Define).min
            if db.ecu_defines.get(ECU_Define).max == None:
                frontRow[4] = '/'
            else:
                frontRow[4] = db.ecu_defines.get(ECU_Define).max
            if db.ecu_defines.get(ECU_Define).defaultValue == '':
                frontRow[5] = '/'
            else:
                db.ecu_defines.get(ECU_Define).defaultValue
        else:
            #print("def : format - xls_common - get_signal - ECU DEF ENUM : {} ".format(db.ecu_defines.get(ECU_Define).values))
            frontRow = ["ECU DEF", ECU_Define, db.ecu_defines.get(ECU_Define).type, ",".join(db.ecu_defines.get(ECU_Define).values),'/','/']
            print("def : format - xls_common - get_signal - ECU DEF ENUM - ",frontRow)
        col = write_excel_line(worksheet_def, row, 0, frontRow, signal_style)
        row += 1

    for FRM_Define in db.frame_defines:
        #print("def : format - xls_common - get_signal - FRM DEF : {} - {} ".format(FRM_Define,db.frame_defines.get(FRM_Define).type))

        if db.frame_defines.get(FRM_Define).type == "INT":

            frontRow = ["FRM DEF", FRM_Define, db.frame_defines.get(FRM_Define).type, db.frame_defines.get(FRM_Define).min, db.frame_defines.get(FRM_Define).max, '/']

        elif db.frame_defines.get(FRM_Define).type == "ENUM":

            print("def : format - xls_common - get_signal - FRM DEF ENUM : {} ".format(",".join(db.frame_defines.get(
                FRM_Define).values)))

            frontRow = ["FRM DEF", FRM_Define, db.frame_defines.get(FRM_Define).type,",".join(db.frame_defines.get(
                FRM_Define).values), '/', '/']
            # print("def : format - xls_common - get_signal - FRM DEF ENUM : {} - {} ".format(FRM_Define, db.frame_defines.get(
            #     FRM_Define).defaultValue))


        # elif db.frame_defines.get(FRM_Define).type == "STRING":


        col = write_excel_line(worksheet_def, row, 0, frontRow, signal_style)
        row += 1

    for SIG_Define in db.signal_defines:
        #print("def : format - xls_common - get_signal - SIG DEF : {} - {} ".format(SIG_Define,db.signal_defines.get(SIG_Define).type))
        #frontRow = ["SIG DEF",SIG_Define,db.signal_defines.get(SIG_Define).type,'/','/','/']

        if db.signal_defines.get(SIG_Define).type == "INT":
            pass

        elif db.signal_defines.get(SIG_Define).type == "ENUM":
            frontRow = ["SIG DEF", SIG_Define, db.signal_defines.get(SIG_Define).type, ",".join(db.signal_defines.get(SIG_Define).values), '/', '/']

        elif db.signal_defines.get(SIG_Define).type == "HEX":
            pass
        elif db.signal_defines.get(SIG_Define).type == "FLOAT":
            frontRow = ["SIG DEF", SIG_Define, db.signal_defines.get(SIG_Define).type, db.signal_defines.get(SIG_Define).min, db.signal_defines.get(SIG_Define).max, '/']

        elif db.signal_defines.get(SIG_Define).type == "STRING":
            frontRow = ["SIG DEF", SIG_Define, db.signal_defines.get(SIG_Define).type, '/', '/', '/']

        col = write_excel_line(worksheet_def, row, 0, frontRow, signal_style)
        row += 1


    # add filter and freeze head_top
    worksheet.autofilter(0, 0, row, len(head_top) - 1)
    worksheet.freeze_panes(1, 0)

    # save file
    workbook.close()


def read_xlsx(file, **args):

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>def : format - xlsx - read_xlsx")

    # type: (typing.Any, **typing.Any) -> typing.Tuple[typing.Dict[typing.Any, str], typing.List[typing.Dict[str, str]]]
    # from: Hooshmand zandi http://stackoverflow.com/a/16544219
    import zipfile
    from xml.etree.ElementTree import iterparse

    sheet = args.get("sheet", 1)
    is_header = args.get("header", False)

    rows = []  # type: typing.List[typing.Dict[str, str]]
    row = {}
    header = {}
    z = zipfile.ZipFile(file)

    # Get shared strings
    strings = [el.text for e, el
               in iterparse(z.open('xl/sharedStrings.xml'))
               if el.tag.endswith('}t')
               ]  # type: typing.List[str]
    value = ''

    # Open specified worksheet
    for e, el in iterparse(z.open('xl/worksheets/sheet%d.xml' % sheet)):
        # get value or index to shared strings
        if el.tag.endswith('}v'):  # <v>84</v>
            value = el.text
        if el.tag.endswith(
                '}c'):  # <c r="A3" t="s"><v>84</v></c>
            # If value is a shared string, use value as an index

            if el.attrib.get('t') == 's':
                value = strings[int(value)]

            # split the row/col information so that the row letter(s) can be separate
            letter = el.attrib['r']  # type: str         # AZ22
            while letter[-1].isdigit():
                letter = letter[:-1]

            # if it is the first row, then create a header hash for the names that COULD be used
            if not rows:
                header[letter] = value.strip()
            else:
                if value != '':
                    # if there is a header row, use the first row's names as the row hash index
                    if is_header is True and letter in header:
                        row[header[letter]] = value
                    else:
                        row[letter] = value

            value = ''
        if el.tag.endswith('}row'):
            rows.append(row)
            row = {}
    z.close()

    return header, rows


def get_if_possible(row, value, default=None):

    print("def : format - xlsx - get_if_possible")

    # type: (typing.Mapping[str, str], str, typing.Optional[str]) -> typing.Union[str, None]
    if value in row:
        return row[value].strip()
    else:
        return default


def load(filename, **options):

    print("def : format - xlsx - load")

    # type: (typing.BinaryIO, **str) -> canmatrix.CanMatrix
    # use xlrd excel reader if available, because its more robust
    if options.get('xlsxLegacy', False) is True:
        logger.error("xlsx: using legacy xlsx-reader - please get xlrd working for better results!")
    else:
        import canmatrix.formats.xls as xls_loader  # we need alias, otherwise we hide the globally imported canmatrix
        return xls_loader.load(filename, **options)
        # else use this hack to read xlsx
    # pip3 install xlrd == 1.2.0 and use xls_loader to load xlsx

    motorola_bit_format = options.get("xlsMotorolaBitFormat", "msbreverse")

    sheet = read_xlsx(filename, sheet=1, header=True)
    db = canmatrix.CanMatrix()
    all_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter_index = list(all_letters)
    letter_index += ["%s%s" % (a, b) for a in all_letters for b in all_letters]

    # Defines not imported...
    #db.add_frame_defines("GenMsgDelayTime", 'INT 0 65535')
    #db.add_frame_defines("GenMsgCycleTimeActive", 'INT 0 65535')
    #db.add_frame_defines("GenMsgNrOfRepetitions", 'INT 0 65535')
    launch_types = []  # type: typing.List[str]

    #db.add_signal_defines("GenSigSNA", 'STRING')

    ecu_start = ecu_end = 0
    if 'Byte Order' in list(sheet[0].values()):
        for key in sheet[0]:
            if sheet[0][key].strip() == 'Byte Order':
                ecu_start = letter_index.index(key) + 1
                break
    else:
        for key in sheet[0]:
            if sheet[0][key].strip() == 'Comment':
                ecu_start = letter_index.index(key) + 1

    for key in sheet[0]:
        if sheet[0][key].strip() == 'Value':
            ecu_end = letter_index.index(-1)

    # ECUs:
    for x in range(ecu_start, ecu_end):
        print("def : format - xlsx - load - ECU : {} - {} - {} - {}".format(
            ecu_start,ecu_end,x,sheet[0][letter_index[x]]))
        db.add_ecu(canmatrix.Ecu(sheet[0][letter_index[x]]))

    # initialize:
    frame_id = None
    signal_name = ""
    signal_length = 8
    new_frame = None  # type: typing.Optional[canmatrix.Frame]
    new_signal = None  # type: typing.Optional[canmatrix.Signal]

    for row in sheet[1]:
        # ignore empty row
        if 'Message ID' not in row:
            continue
        # new frame detected
        if row['Message ID'] != frame_id:
            # new Frame
            frame_id = row['Message ID']
            frame_name = row['Message Name']
            cycle_time = row['Period [ms]']
            # launch_type = get_if_possible(row, 'Launch Type')
            # dlc = 8
            dlc = get_if_possible(row, 'DLC [Byte]')
            # launch_param = get_if_possible(row, 'Launch Parameter', '0')
            # launch_param = str(int(launch_param))
            '''
            if frame_id.endswith("xh"):
                new_frame = canmatrix.Frame(frame_name, arbitration_id=int(frame_id[:-2], 16), size=dlc)
                new_frame.arbitration_id.extended = True
            else:
                new_frame = canmatrix.Frame(frame_name, arbitration_id=int(frame_id[:-1], 16), size=dlc)
            '''
            if frame_id.startsswith("0xC"):
                new_frame = canmatrix.Frame(frame_name, arbitration_id=int(frame_id[:-2], 16), size=dlc)
                new_frame.arbitration_id.extended = True
            else:
                new_frame = canmatrix.Frame(frame_name, arbitration_id=int(frame_id[:-1], 16), size=dlc)

            db.add_frame(new_frame)
            '''
            # eval launch_type
            if launch_type is not None:
                new_frame.add_attribute("GenMsgSendType", launch_type)
                if launch_type not in launch_types:
                    launch_types.append(launch_type)
            '''
            new_frame.cycle_time = cycle_time
        # new signal detected
        if 'Signal Name' in row and row['Signal Name'] != signal_name:
            receiver = []  # type: typing.List[str]
            # start_byte = int(row["Signal Byte No."])
            # start_bit = int(row['Signal Bit No.'])
            signal_name = row['Signal Name']
            signal_comment = get_if_possible(row, 'Comment')
            signal_length = int(row['size [Bit]'])
            # signal_default = get_if_possible(row, 'Signal Default')
            # signal_sna = get_if_possible(row, 'Signal Not Available')
            # multiplex = None  # type: typing.Union[str, int, None]
            '''
            if signal_comment is not None and signal_comment.startswith('Mode Signal:'):
                multiplex = 'Multiplexor'
                signal_comment = signal_comment[12:]
            elif signal_comment is not None and signal_comment.startswith('Mode '):
                mux, signal_comment = signal_comment[4:].split(':', 1)
                multiplex = int(mux.strip())
            '''
            signal_byte_order = get_if_possible(row, 'Byte Order')
            if signal_byte_order is not None:
                if 'intel' in signal_byte_order:
                    is_little_endian = True
                else:
                    is_little_endian = False
            else:
                is_little_endian = True  # Default Intel

            is_signed = False

            if signal_name != "-":
                for x in range(ecu_start, ecu_end):
                    ecu_name = sheet[0][letter_index[x]].strip()
                    ecu_sender_receiver = get_if_possible(row, ecu_name)
                    if ecu_sender_receiver is not None:
                        if 's' in ecu_sender_receiver:
                            new_frame.add_transmitter(ecu_name)
                        if 'r' in ecu_sender_receiver:
                            receiver.append(ecu_name)
                new_signal = canmatrix.Signal(signal_name,
                                              start_bit=(start_byte - 1) * 8 + start_bit,
                                              size=signal_length,
                                              is_little_endian=is_little_endian,
                                              is_signed=is_signed,
                                              receivers=receiver,
                                              multiplex=multiplex)

                if not is_little_endian:
                    # motorola
                    if motorola_bit_format == "msb":
                        new_signal.set_startbit(
                            (start_byte - 1) * 8 + start_bit, bitNumbering=1)
                    elif motorola_bit_format == "msbreverse":
                        new_signal.set_startbit((start_byte - 1) * 8 + start_bit)
                    else:  # motorola_bit_format == "lsb"
                        new_signal.set_startbit(
                            (start_byte - 1) * 8 + start_bit,
                            bitNumbering=1,
                            startLittle=True
                        )
                new_frame.add_signal(new_signal)
                new_signal.add_comment(signal_comment)

    # dlc-estimation / dlc is not in xls, thus calculate a minimum-dlc:
    for frame in db.frames:
        frame.update_receiver()
        frame.calc_dlc()

    launch_type_enum = "ENUM"
    for launch_type in launch_types:
        if len(launch_type) > 0:
            launch_type_enum += ' "' + launch_type + '",'
    db.add_frame_defines("GenMsgSendType", launch_type_enum[:-1])

    db.set_fd_type()
    return db
