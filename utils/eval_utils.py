def remove_none_slots(belief):
    for slot_tuple in belief:
        domain, slot_name, slot_value = slot_tuple.split('-')
        if slot_value == 'none':
            continue
        if slot_value == "do n't care":
            yield '-'.join((domain, slot_name, 'dontcare'))
            continue

        # HACK HACK HACK
        # if domain != 'train':
        #    continue

        yield slot_tuple


def get_joint_accuracy(turn):
    return float(set(remove_none_slots(turn['turn_belief'])) == set(remove_none_slots(turn['pred_bs_ptr'])))


def print_turn(dialogue_dev_data, pred_data, dialogue_id, up_to=-1):
    dialogue = dialogue_dev_data[dialogue_id]
    if up_to < 0:
        up_to = len(dialogue['dialogue'])

    print()
    print(dialogue_id + '/' + str(up_to))
    for turn_idx in range(up_to + 1):
        if turn_idx > 0:
            print('S: ' + dialogue['dialogue'][turn_idx]['system_transcript'])
        print('U: ' + dialogue['dialogue'][turn_idx]['transcript'])
    pred_data[dialogue_id][str(up_to)]['turn_belief'].sort()
    pred_data[dialogue_id][str(up_to)]['pred_bs_ptr'].sort()
    print('Ann:', pred_data[dialogue_id][str(up_to)]['turn_belief'])
    print('Pred:', pred_data[dialogue_id][str(up_to)]['pred_bs_ptr'])


def get_slot_names(belief):
    for slot_tuple in belief:
        domain, slot_name, slot_value = slot_tuple.split('-')
        if slot_value == 'none':
            continue
        yield domain + '-' + slot_name


def get_name_accuracy(annotation, prediction):
    return float(set(get_slot_names(annotation)) == set(get_slot_names(prediction)))
