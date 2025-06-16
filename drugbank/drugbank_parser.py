# # https://caseolap.github.io/docs/drug/drugbank/
# #https://gist.github.com/rosherbal/56461421c69a8a7da775336c95fa62e0

import os
import zipfile
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XMLParser, XMLPullParser
import pandas as pd
import xmlschema
import json as json
import sys
import unicodedata
import re
from importlib import resources as impresources
from ddi_fw.utils import ZipHelper


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def replace_key(key: str):
    if key.startswith('@'):
        key = key[1:]
    if key == '$':
        key = "value"
    elif '{http://www.drugbank.ca}' in key:
        key = key.replace('{http://www.drugbank.ca}', '')
    return key


def modify_keys(d):
    for k, v in d.copy().items():
        if isinstance(v, dict):
            d.pop(k)
            d[replace_key(k)] = v
            modify_keys(v)
        elif isinstance(v, list):
            d.pop(k)
            d[replace_key(k)] = v
            for i in v:
                if isinstance(i, list) or isinstance(i, dict):
                    modify_keys(i)
                # print(i)

        else:
            if k == "keyToChange":
                v = int(v)
            d.pop(k)
            d[replace_key(k)] = v
    return d


class DrugBankParser:
    def __init__(self, zip_file='drugbank.zip', input_path='./drugbank'):

        # sys.path.insert(0,'/content/drive/My Drive/drugbank')
        # HERE = '/content/drive/My Drive/drugbank'
        HERE = input_path
        xsd_file='drugbank.xsd'
        DRUGBANK_XSD = impresources.files("ddi_fw.drugbank").joinpath("drugbank.xsd").open()
        # DRUGBANK_XSD = HERE + '/' + xsd_file
        DRUGBANK_ZIP = HERE + '/' + zip_file
        xsd = xmlschema.XMLSchema(DRUGBANK_XSD)
        self.drug_type_schema = xsd.complex_types[1]
        self.zf = zipfile.ZipFile(DRUGBANK_ZIP, 'r')

    def parse(self, save_path='./drugbank/drugs', override = False):
        if not override:
            print('No parsing process has been executed!!!')
            return

        elements = []
        k = 0

        for name in self.zf.namelist():
            f = self.zf.open(name)
            # tree = ET.parse(f)
            # root = tree.getroot()
            previous_element = None
            for event, element in ET.iterparse(f, events=('end',)):  # "end"
                # if k == 10:
                #     break
                if len(elements) == 0:
                    elements.append(element)
                elif len(elements) == 1:
                    elements.append(element)
                elif len(elements) == 2:
                    elements[0] = elements[1]
                    elements[1] = element
                if len(elements) == 2:
                    previous_element = elements[len(elements)-2]
                drug = None
                # previous_element = element.find("..")
                #
                if previous_element is not None and previous_element.tag == '{http://www.drugbank.ca}transporters' and event == 'end' and element.tag == "{http://www.drugbank.ca}drug":
                    drug = element
                    elements = []

                    # for child in element:
                    #     print(child.text)

                if drug is None:
                    continue

                name = drug.find("{http://www.drugbank.ca}name")

                d_name = None
                if name is not None:
                    d_name = name.text
                    line = name.text

                if d_name is None:
                    continue

                k = k + 1

                # print(d_name)

                # if lax is used we have to send d[0] as a parameter
                d = self.drug_type_schema.decode(drug, validation='strict')
                # pretty_dict = {replace_key(k): v for k, v in d[0].items()}
                pretty_dict = modify_keys(d)
                # for key, value in pretty_dict.items():
                #     print(key, '->', value)
                # file_name = slugify(d_name)

                from pathlib import Path

                Path(save_path).mkdir(parents=True, exist_ok=True)

                primary_id = [
                    id['value'] for id in pretty_dict["drugbank-id"] if id['primary'] == True][0]
                with open(f'{save_path}/{primary_id}.json', 'w', encoding='utf-8') as f:
                    json.dump(pretty_dict, f, ensure_ascii=False, indent=4)

        print("Done")

    def zip_files(self, chunk_size=1000, input_path='./drugbank/drugs', output_path='./drugbank/zips'):
        zip_helper = ZipHelper()
        zip_helper.zip(zip_prefix='drugs', input_path=input_path,
                       output_path=output_path, chunk_size=chunk_size)

