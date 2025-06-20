import sqlite3
import pandas as pd
import os
import json
import glob
from tqdm import tqdm

import csv

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from drugbank.event_extractor import EventExtractor
from ddi_fw.utils import ZipHelper

def multiline_to_singleline(multiline):
    if multiline is None:
        return ""
    return " ".join(line.strip() for line in multiline.splitlines())

# targets -> target -> polypeptide
# enzymes -> enzyme -> polypeptide
# pathways from KEGG, KEGG ID is obtained from DrugBank
# https://www.genome.jp/dbget-bin/www_bget?drug:D03136
# https://www.kegg.jp/entry/D03136


class DrugBankProcessor():

    def mask_interaction(self, drug_1, drug_2, interaction):
        return interaction.replace(
            drug_1, "DRUG").replace(drug_2, "DRUG")

    def extract_zip_files(self, input_path='zips', output_path='drugs', override=False):
        if override:
            zip_helper = ZipHelper()
            zip_helper.extract(input_path=input_path, output_path=output_path)

    def get_external_identifiers(self, input_path='drugs'):
        external_identifier_list = []
        all_json_files = input_path+'/*.json*'

        for filepath in tqdm(glob.glob(all_json_files)):
            with open(filepath, 'r', encoding="utf8") as f:

                data = json.load(f)
                drug_1 = data['name']
                drug_1_id = [d['value']
                             for d in data['drugbank-id'] if d['primary'] == True][0]
                external_identifiers = data['external-identifiers'] if "external-identifiers" in data else None
                external_identifiers_dict = {}
                external_identifiers_dict['name'] = drug_1
                external_identifiers_dict['drugbank_id'] = drug_1_id
                if external_identifiers is not None:
                    for p in external_identifiers['external-identifier']:
                        external_identifiers_dict[p['resource'].lower().replace(
                            " ", "_")] = p['identifier']
                    # external_identifiers_dict = dict(
                    #     [(p['resource'].lower().replace(" ","_"), p['identifier']) for p in external_identifiers['external-identifier']])
                    # external_identifiers_dict['name'] = drug_1
                    # external_identifiers_dict['drugbank_id'] = drug_1_id
                external_identifier_list.append(external_identifiers_dict)
        return external_identifier_list

    def process(self,
                input_path='drugs',
                output_path='output',
                db_path=r"./drugbank.db"):


        drug_rows = []
        all_ddis = []
        external_identifier_list = []
        all_json_files = input_path+'/*.json*'

        for filepath in tqdm(glob.glob(all_json_files)):
            with open(filepath, 'r', encoding="utf8") as f:

                data = json.load(f)

                drug_1 = data['name']
                drug_1_id = [d['value']
                             for d in data['drugbank-id'] if d['primary'] == True][0]
                description = multiline_to_singleline(
                    data['description'])
                if data['drug-interactions'] is not None:
                    drug_interactions = [
                        interaction for interaction in data['drug-interactions']['drug-interaction']]
                    ddis = [(drug_1, interaction['name'], interaction['description'])
                            for interaction in data['drug-interactions']['drug-interaction']]

                    ddi_dict = [{
                        'drug_1_id': drug_1_id,
                        'drug_1': drug_1,
                        'drug_2_id': interaction['drugbank-id']['value'],
                        'drug_2': interaction['name'],
                        'interaction': interaction['description'],
                        'masked_interaction': self.mask_interaction(drug_1, interaction['name'], interaction['description'])}
                        for interaction in data['drug-interactions']['drug-interaction']]
                    all_ddis.extend(ddi_dict)

                synthesis_reference = data['synthesis-reference']
                indication = multiline_to_singleline(
                    data['indication'])
                pharmacodynamics = multiline_to_singleline(
                    data['pharmacodynamics'])
                mechanism_of_action = multiline_to_singleline(
                    data['mechanism-of-action'])
                toxicity = multiline_to_singleline(data['toxicity'])
                metabolism = multiline_to_singleline(
                    data['metabolism'])
                absorption = multiline_to_singleline(
                    data['absorption'])
                half_life = multiline_to_singleline(data['half-life'])
                protein_binding = multiline_to_singleline(
                    data['protein-binding'])
                route_of_elimination = multiline_to_singleline(
                    data['route-of-elimination'])
                volume_of_distribution = multiline_to_singleline(
                    data['volume-of-distribution'])
                clearance = multiline_to_singleline(data['clearance'])

                food_interactions = data['food-interactions']
                sequences = data['sequences'] if "sequences" in data else None

                external_identifiers = data['external-identifiers'] if "external-identifiers" in data else None
                experimental_properties = data['experimental-properties'] if "experimental-properties" in data else None
                calculated_properties = data['calculated-properties'] if "calculated-properties" in data else None

                enzymes_polypeptides = None
                targets_polypeptides = None
                pathways = None

                # targets = data['targets'] if "targets" in data else None
                if data['targets'] is not None:
                    # targets_polypeptides = [p['id'] for d in data['targets']['target'] for p in d['polypeptide'] if 'polypeptide' in d ]
                    targets_polypeptides = [
                        p['id'] for d in data['targets']['target'] if 'polypeptide' in d for p in d['polypeptide']]

                if data['enzymes'] is not None:
                    # enzymes_polypeptides = [p['id'] for d in data['enzymes']['enzyme'] for p in d['polypeptide'] if 'polypeptide' in d]
                    enzymes_polypeptides = [
                        p['id'] for d in data['enzymes']['enzyme'] if 'polypeptide' in d for p in d['polypeptide']]

                if data['pathways'] is not None:
                    pathways = [
                        d['smpdb-id'] for d in data['pathways']['pathway']]

                if external_identifiers is not None:
                    external_identifiers_dict = dict(
                        [(p['resource'], p['identifier']) for p in external_identifiers['external-identifier']])
                    external_identifiers_dict['drugbank_id'] = drug_1_id
                    external_identifier_list.append(
                        external_identifiers_dict)
                smiles = None
                morgan_hashed = None
                if calculated_properties is not None:
                    calculated_properties_dict = dict(
                        [(p['kind'], p['value']) for p in calculated_properties['property']])
                    smiles = calculated_properties_dict['SMILES'] if 'SMILES' in calculated_properties_dict else None
                    if smiles is not None:
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(
                                mol, 2, nBits=881).ToList()
                        except:
                            print("An exception occurred")
                if morgan_hashed is None:
                    morgan_hashed = np.zeros(881).tolist()

                row = {'drugbank_id': drug_1_id,
                       'name': drug_1,
                       'description': description,
                       'synthesis_reference': synthesis_reference,
                       'indication': indication,
                       'pharmacodynamics': pharmacodynamics,
                       'mechanism_of_action': mechanism_of_action,
                       'toxicity': toxicity,
                       'metabolism': metabolism,
                       'absorption': absorption,
                       'half_life': half_life,
                       'protein_binding': protein_binding,
                       'route_of_elimination': route_of_elimination,
                       'volume_of_distribution': volume_of_distribution,
                       'clearance': clearance,
                       'smiles': smiles,
                       'smiles_morgan_fingerprint': ','.join(map(str, morgan_hashed)),
                       'enzymes_polypeptides': '|'.join(enzymes_polypeptides) if enzymes_polypeptides is not None else None,
                       'targets_polypeptides': '|'.join(targets_polypeptides) if targets_polypeptides is not None else None,
                       'pathways': '|'.join(pathways) if pathways is not None else None
                       #    'external_identifiers': external_identifiers_dict
                       }
                drug_rows.append(row)

        drug_names = ['DRUG']
        event_extractor = EventExtractor(drug_names)

        # replace_dict = {'MYO-029': 'Stamulumab'}
        # for ddi in tqdm(all_ddis):
        #     for key, value in replace_dict.items():
        #         ddi['masked_interaction'] = ddi['masked_interaction'].replace(
        #             key, value)

        self.drugs_df = pd.DataFrame(drug_rows)
        self.ddis_df = pd.DataFrame(all_ddis)

        count = [0]

        def fnc2(interaction, count):
            count[0] = count[0] + 1
            if count[0] % 1000 == 0:
                print(f'{count[0]}/{len(all_ddis)}')
            mechanism, action, drugA, drugB = event_extractor.extract(
                interaction)
            return mechanism+'__' + action

        self.ddis_df['mechanism_action'] = self.ddis_df['masked_interaction'].apply(
            fnc2, args=(count,))

        zip_helper = ZipHelper()

        conn = sqlite3.connect(db_path)
        self.drugs_df.to_sql(
            '_Drugs', conn, if_exists='replace', index=True)
        self.ddis_df.to_sql('_Interactions', conn,
                            if_exists='replace', index=True)
        ext_id_df = pd.DataFrame.from_records(external_identifier_list)
        ext_id_df.to_sql('_ExternalIdentifiers', conn,
                         if_exists='replace', index=True)

        zip_helper.zip_single_file(zip_name='db',
                                   file_path=db_path, output_path=output_path+'/zips')
        conn.close()
