NUM_CLASSES = 15

def get_instance_count():
    instance_count = [None] * NUM_CLASSES
    for i, c in enumerate(NDL_CATEGORIES):
        instance_count[i] = c['instance_count']
    return instance_count

NDL_CATEGORIES = [
    {'id':  0, 'name': 'line_main',    'instance_count': 100},
    {'id':  1, 'name': 'line_inote',   'instance_count': 100},
    {'id':  2, 'name': 'line_hnote',   'instance_count': 1},
    {'id':  3, 'name': 'line_caption', 'instance_count': 1},
    {'id':  4, 'name': 'block_fig',    'instance_count': 100},
    {'id':  5, 'name': 'block_table',  'instance_count': 1},
    {'id':  6, 'name': 'block_pillar', 'instance_count': 100},
    {'id':  7, 'name': 'block_folio',  'instance_count': 100},
    {'id':  8, 'name': 'block_rubi',   'instance_count': 100},
    {'id':  9, 'name': 'block_chart',  'instance_count': 1},
    {'id': 10, 'name': 'block_eqn',    'instance_count': 1},
    {'id': 11, 'name': 'block_cfm',    'instance_count': 1},
    {'id': 12, 'name': 'block_eng',    'instance_count': 1},
    {'id': 13, 'name': 'char',         'instance_count': 1},
    {'id': 14, 'name': 'void',         'instance_count': 1}
]