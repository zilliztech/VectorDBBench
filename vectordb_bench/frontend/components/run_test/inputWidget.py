from vectordb_bench.frontend.config.dbCaseConfigs import CaseConfigInput, InputType


def inputWidget(st, config: CaseConfigInput, key: str):
    if config.inputType == InputType.Text:
        return st.text_input(
            config.displayLabel if config.displayLabel else config.label.value,
            key=key,
            help=config.inputHelp,
            value=config.inputConfig["value"],
        )
    if config.inputType == InputType.Option:
        return st.selectbox(
            config.displayLabel if config.displayLabel else config.label.value,
            config.inputConfig["options"],
            key=key,
            help=config.inputHelp,
        )
    if config.inputType == InputType.Number:
        return st.number_input(
            config.displayLabel if config.displayLabel else config.label.value,
            # format="%d",
            step=config.inputConfig.get("step", 1),
            min_value=config.inputConfig["min"],
            max_value=config.inputConfig["max"],
            key=key,
            value=config.inputConfig["value"],
            help=config.inputHelp,
        )
    if config.inputType == InputType.Float:
        return st.number_input(
            config.displayLabel if config.displayLabel else config.label.value,
            step=config.inputConfig.get("step", 0.1),
            min_value=config.inputConfig["min"],
            max_value=config.inputConfig["max"],
            key=key,
            value=config.inputConfig["value"],
            help=config.inputHelp,
        )
    if config.inputType == InputType.Bool:
        return st.selectbox(
            config.displayLabel if config.displayLabel else config.label.value,
            options=[True, False],
            index=0 if config.inputConfig["value"] else 1,
            key=key,
            help=config.inputHelp,
        )
    raise Exception(f"Invalid InputType: {config.inputType}")
