def get_group_offsets(group_names, offset_scale=0.12):
    if len(group_names) == 1:
        return {group_names[0]: 0}

    offset_positions = [
        index - (len(group_names) - 1) / 2 for index in range(len(group_names))
    ]
    return {
        group_name: position * offset_scale
        for group_name, position in zip(group_names, offset_positions)
    }
