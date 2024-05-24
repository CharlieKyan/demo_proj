# from kg import KnowledgeGraph, Entity, EntityResponse
# import json

# kg = KnowledgeGraph("datasets/slake/KG", ["en_disease", "en_organ_rel", "en_organ"])
# splits = ['train', 'validate', 'test']
# for split in splits:
#     with open(f'datasets/slake/{split}_preprocessed.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     for entry in data:
#         question