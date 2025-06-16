'''
copied from https://github.com/YifanDengWHU/DDIMDL/blob/master/NLPProcess.py and reorganized
'''

# import stanfordnlp
# stanfordnlp.download("en")
import pandas as pd
import stanza
# stanza.download("en")

import numpy as np


class EventExtractor:
    def __init__(self, druglist, use_cache=True):
        self.druglist = druglist
        self.druglist2 = ['_'.join(d.replace('.', ' ').replace(
            ',', ' ').replace('-', ' ').split(' ')) for d in druglist]
        # self.events = events
        self.pipeline = stanza.Pipeline(use_gpu=True)
        self.cache = dict()

    def prepare_event_text(self, event):
        for ex, new in zip(self.druglist, self.druglist2):
            event = event.replace(ex, new)
        return event

    def extract_all(self, events):
        mechanisms = []
        actions = []
        drugA_list = []
        drugB_list = []
        for i in range(len(events)):
            mechanism, action, drugA, drugB = self.extract(events[i])
            mechanisms.append(mechanism)
            actions.append(action)
            drugA_list.append(drugA)
            drugB_list.append(drugB)
        return mechanisms, actions, drugA_list, drugB_list

    def extract(self, event):
        if event in self.cache:
            return self.cache[event]
        event = self.prepare_event_text(event)
        drugA = None
        drugB = None

        def addMechanism(node):
            if int(sonsNum[int(node-1)]) == 0:
                return
            else:
                for k in sons[node-1]:
                    if int(k) == 0:
                        break
                    if dependency[int(k - 1)].text == drugA or dependency[int(k - 1)].text == drugB:
                        continue
                    quene.append(int(k))
                    addMechanism(int(k))
            return quene

        doc = self.pipeline(event)
        dependency = []
        for j in range(len(doc.sentences[0].words)):
            dependency.append(doc.sentences[0].words[j])
        sons = np.zeros((len(dependency), len(dependency)))
        sonsNum = np.zeros(len(dependency))
        flag = False
        count = 0
        for j in dependency:
            # if j.dependency_relation=='root':
            if j.deprel == 'root':
                # root=int(j.index)
                root = int(j.id)
                action = j.lemma
            if j.text in self.druglist2:
                if count < 2:
                    if flag == True:
                        drugB = j.text
                        count += 1
                    else:
                        drugA = j.text
                        flag = True
                        count += 1
            sonsNum[j.head-1] += 1
            sons[j.head-1, int(sonsNum[j.head-1]-1)] = int(j.id)
        quene = []
        for j in range(int(sonsNum[root-1])):
            if dependency[int(sons[root-1, j]-1)].deprel == 'obj' or dependency[int(sons[root-1, j]-1)].deprel == 'nsubj:pass':
                quene.append(int(sons[root-1, j]))
                break
        quene = addMechanism(quene[0])
        quene.sort()

        mechanism = " ".join(dependency[j-1].text for j in quene)
        if mechanism == "the fluid retaining activities":
            mechanism = "the fluid"
        if mechanism == "atrioventricular blocking ( AV block )":
            mechanism = 'the atrioventricular blocking ( AV block ) activities increase'

        self.cache[event] = (mechanism, action,
                             drugA.replace('_', ' ') if drugA != None else '',
                             drugB.replace('_', ' ') if drugB != None else '')
        

        if drugA == '' or drugB == '':
            print(event)
 
        return mechanism, action, drugA.replace('_', ' ') if drugA != None else '',  drugB.replace('_', ' ') if drugB != None else ''


# drugs_pickle_path = 'drugbank/output/drugs.pkl'
# drugs_df = pd.read_pickle(drugs_pickle_path)

# drug_names = drugs_df['name'].to_list()


# drug_names = ['Lepirudin','Ursodeoxycholic acid']
# event_extractor = EventExtractor(
#     drug_names)

# mechanisms, actions, drugA_list, drugB_list = event_extractor.extract_all(
#     ['The risk or severity of bleeding and bruising can be increased when Lepirudin is combined with Ursodeoxycholic acid'])
# # mechanism, action, drugA, drugB = event_extractor.extract(
# #     'Bivalirudin may increase the anticoagulant activities of Bromfenac')


# print(mechanisms)
