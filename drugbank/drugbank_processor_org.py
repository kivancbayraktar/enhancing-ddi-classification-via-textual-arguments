import pandas as pd
import os
import json
import glob
from tqdm import tqdm

import csv

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from ddi_fw.drugbank.event_extractor import EventExtractor

from zip_helper import ZipHelper
# from event_extractor import EventExtractor


def multiline_to_singleline(multiline):
    if multiline is None:
        return ""
    return " ".join(line.strip() for line in multiline.splitlines())

# targets -> target -> polypeptide
# enzymes -> enzyme -> polypeptide
# pathways from KEGG, KEGG ID is obtained from ddi_fw.drugbank
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

    def process(self, input_path='drugs', output_path='output', zip_outputs=True):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        drugs_pickle_path = output_path+'/drugs.pkl'
        drugs_csv_path = output_path+'/drugs.gzip'
        ddi_pickle_path = output_path + '/ddi.pkl'
        ddi_csv_path = output_path + '/ddi.gzip'

        if not os.path.exists(drugs_pickle_path) or not os.path.exists(ddi_pickle_path):
            drug_rows = []
            all_ddis = []
            all_json_files = input_path+'/*.json*'

            for filepath in tqdm(glob.glob(all_json_files)):
                with open(filepath, 'r', encoding="utf8") as f:

                    data = json.load(f)

                    # if data['drug-interactions'] is None:
                    if False:
                        continue
                    else:
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

                        # targets = data['targets'] if "targets" in data else None
                        if data['targets'] is not None:
                            # targets_polypeptides = [p['id'] for d in data['targets']['target'] for p in d['polypeptide'] if 'polypeptide' in d ]
                            targets_polypeptides = [
                                p['id'] for d in data['targets']['target'] if 'polypeptide' in d for p in d['polypeptide']]

                        if data['enzymes'] is not None:
                            # enzymes_polypeptides = [p['id'] for d in data['enzymes']['enzyme'] for p in d['polypeptide'] if 'polypeptide' in d]
                            enzymes_polypeptides = [
                                p['id'] for d in data['enzymes']['enzyme'] if 'polypeptide' in d for p in d['polypeptide']]

                        if external_identifiers is not None:
                            external_identifiers_dict = dict(
                                [(p['resource'], p['identifier']) for p in external_identifiers['external-identifier']])

# add note column
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
                            morgan_hashed = np.zeros(881)

                        # k = [p[k]  for p in calculated_properties['property'] for k in p.keys() if k =='SMILES']
                        # external_identifiers['external-identifier']
                        # experimental_properties['property']

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
                               'smiles_morgan_fingerprint': morgan_hashed,
                               'enzymes_polypeptides': enzymes_polypeptides,
                               'targets_polypeptides': targets_polypeptides,
                               'external_identifiers': external_identifiers_dict
                               }
                        drug_rows.append(row)

                    # if len(drug_rows) == 10:
                    #     break
            # print(smiles_count)
            print(f"Size of drugs {len(drug_rows)}")
            print(f"Size of DDIs {len(all_ddis)}")
            np.set_printoptions(threshold=np.inf)

            # drug_names = [row['name'] for row in drug_rows]
            drug_names = ['DRUG']
            event_extractor = EventExtractor(drug_names)

            replace_dict = {'MYO-029': 'Stamulumab'}
            for ddi in tqdm(all_ddis):
                for key, value in replace_dict.items():
                    ddi['masked_interaction'] = ddi['masked_interaction'].replace(
                        key, value)
            #     interaction = ddi['interaction']
            #     mechanism, action, drugA, drugB = event_extractor.extract(interaction)
            #     ddi['mechanism'] = mechanism
            #     ddi['action'] = action

            self.drugs_df = pd.DataFrame(drug_rows)
            self.drugs_df.to_pickle(drugs_pickle_path)
            self.drugs_df.to_csv(
                drugs_csv_path,  index=False, compression='gzip')

            # print('mechanism_action calculation')
            self.ddis_df = pd.DataFrame(all_ddis)

            count = [0]

            def fnc2(interaction, count):
                count[0] = count[0] + 1
                if count[0] % 1000 == 0:
                    print(f'{count[0]}/{len(all_ddis)}')
                mechanism, action, drugA, drugB = event_extractor.extract(
                    interaction)
                return mechanism+'__' + action

            # self.ddis_df['mechanism_action'] = self.ddis_df['interaction'].apply(lambda x: fnc2(x))
            # tqdm.pandas()
            self.ddis_df['mechanism_action'] = self.ddis_df['masked_interaction'].apply(
                fnc2, args=(count,))

            self.ddis_df.to_csv(ddi_csv_path,  index=False, compression='gzip')
            self.ddis_df.to_pickle(ddi_pickle_path)

            if zip_outputs:
                zip_helper = ZipHelper()
                zip_helper.zip_single_file(
                    file_path=drugs_pickle_path, output_path=output_path+'/zips', name='drugs-pickle')
                zip_helper.zip_single_file(
                    file_path=ddi_pickle_path, output_path=output_path+'/zips', name='ddi-pickle')

        else:
            print('Output path has processed data, load function is called')
            self.load(output_path)

    def load(self, path):
        drugs_pickle_path = path+'/drugs.pkl'
        ddi_pickle_path = path+'/ddi.pkl'
        if os.path.exists(drugs_pickle_path) and os.path.exists(ddi_pickle_path):
            self.drugs_df = pd.read_pickle(drugs_pickle_path)
            self.ddis_df = pd.read_pickle(ddi_pickle_path)
        else:
            print('One of given paths could not found')

    def load_from_csv(self, path):
        drugs_csv_path = path+'/drugs.gzip'
        ddi_csv_path = path+'/ddi.gzip'
        if os.path.exists(drugs_csv_path) and os.path.exists(ddi_csv_path):
            self.drugs_df = pd.read_csv(drugs_csv_path, compression='gzip')
            self.ddis_df = pd.read_csv(ddi_csv_path, compression='gzip')
        else:
            print('One of given paths could not found')

    def load2(self, path):
        drugs_pickle_path = path+'/drugs.pkl'
        ddi_csv_path = path+'/ddi.gzip'
        if os.path.exists(drugs_pickle_path) and os.path.exists(ddi_csv_path):
            self.drugs_df = pd.read_pickle(drugs_pickle_path)
            self.ddis_df = pd.read_csv(ddi_csv_path, compression='gzip')
        else:
            print('One of given paths could not found')

    def drugs_as_dataframe(self):
        return self.drugs_df

    def filtered_drugs_as_dataframe(self, drug_ids):
        return self.drugs_df[self.drugs_df['drugbank_id'].isin(drug_ids)]

    def ddis_as_dataframe(self):
        return self.ddis_df

    def filtered_ddis(self, drugs):
        ddis_df = self.ddis_df.copy()
        return ddis_df[(ddis_df['drug_1'] in drugs) & (
            ddis_df['drug_2'] in drugs)]
