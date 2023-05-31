def getShowResults(results, st):
    st.header("Results")
    resultSelectOptions = [
        result.task_label if result.task_label != result.run_id else f"res-{result.run_id[:4]}"
        for i, result in enumerate(results)
    ]
    selectedResultSelectedOptions = st.multiselect(
        "results",
        resultSelectOptions,
        label_visibility="hidden",
        default=resultSelectOptions,
    )
    selectedResult = []
    for option in selectedResultSelectedOptions:
        result = results[resultSelectOptions.index(option)].results
        selectedResult += result

    return selectedResult