# TODO: remove xlsxwriter import?
import xlsxwriter # library to write to excel
import pandas as pd # data science library
import os

class Excel():

    def __init__(self, parameterResults: pd.DataFrame) -> None:
        self.parameterResults = parameterResults
        self.setupDir()

    def setupDir(self) -> None:
        # make directory: data
        try:
            os.mkdir(f'data')
        except OSError as error:
            pass

    def write(self) -> None:
        # merge dataframes to combine into one excel document
        dataframes = []
        for ticker in self.parameterResults.keys():
            dataframe = self.parameterResults[ticker]
            dataframes.append(dataframe)
        mergedDataFrame = pd.concat(dataframes)
    
        # begin writing
        writer = pd.ExcelWriter('data/dca-parameters.xlsx', engine = 'xlsxwriter')
        mergedDataFrame.to_excel(writer, sheet_name = 'DCA Parameters', index = False)

        font_color = '#ffffff'
        background_color = '#000000'

        USE_BORDER = False
        string_format = writer.book.add_format(
            {
                # 'font_color': font_color,
                # 'bg_color': background_color,
                'border': 1 if USE_BORDER else 0,
                'center_across': True
            }
        )

        dollar_format = writer.book.add_format(
            {
                'num_format': '$#,##0.00',
                # 'font_color': font_color,
                # 'bg_color': background_color,
                'border': 1 if USE_BORDER else 0,
                'center_across': True
            }
        )

        integer_format = writer.book.add_format(
            {
                'num_format': '#,##0',
                # 'font_color': font_color,
                # 'bg_color': background_color,
                'border': 1 if USE_BORDER else 0,
                'center_across': True
            }
        )

        float_format = writer.book.add_format(
            {
                'num_format': '#,##0.00',
                # 'font_color': font_color,
                # 'bg_color': background_color,
                'border': 1 if USE_BORDER else 0,
                'center_across': True
            }
        )

        yes_no_format = writer.book.add_format(
            {
                # 'font_color': font_color,
                # 'bg_color': background_color,
                'border': 1 if USE_BORDER else 0,
                'center_across': True
            }
        )

        # used to dynamically allocate column formats
        ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # loop through columns and set format
        column_formats = {}
        for i in range(len(mergedDataFrame.columns)):
            # dynamically setup column formats
            column_formats[ALPHABET[i % 26] * ((i // 26) + 1)] = [mergedDataFrame.columns[i],
                string_format if mergedDataFrame.columns[i] == 'Ticker'
                else yes_no_format
            ]

        # write data to excel document with correct formats
        for column in column_formats.keys():
            writer.sheets['DCA Parameters'].set_column(f'{column}:{column}', len(column_formats[column][0]), column_formats[column][1])
            writer.sheets['DCA Parameters'].write(f'{column}1', column_formats[column][0], string_format)

        # save excel document
        writer.save()