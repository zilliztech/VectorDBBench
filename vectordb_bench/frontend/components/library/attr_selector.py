def attr_selector(st, options, label, data, default_attr=None):
    labels = st.multiselect(label, options, default=default_attr)
    if len(labels) == 0:
        return None
    if len(labels) == 1:
        return labels[0]

    attrs = [label for label in labels]
    attrs.sort(key=lambda x: 0 if "index" in x else 1)
    new_attr = "-".join(attrs)
    for item in data:
        valid_attr_value = ""
        for attr in attrs:
            v = f'{item.get(attr, "")}'
            if v:
                if v.isdigit():
                    if len(valid_attr_value) > 0:
                        valid_attr_value += ","
                    valid_attr_value += f"{attr}={v}"
                else:
                    if len(valid_attr_value) > 0:
                        valid_attr_value += ","
                    valid_attr_value += f"{v}"
        if valid_attr_value.count(',') >= 2:
            valid_attr_value = valid_attr_value.replace(',', '<br>')
        item[new_attr] = valid_attr_value
    return new_attr
