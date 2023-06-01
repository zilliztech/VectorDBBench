def getShowResults(results, st):
    st.subheader("Results")
    resultSelectOptions = [
        result.task_label
        if result.task_label != result.run_id
        else f"res-{result.run_id[:4]}"
        for i, result in enumerate(results)
    ]
    if len(resultSelectOptions) == 0:
        st.write(
            "There are no results to display. Please wait for the task to complete or run a new task."
        )
        return []

    selectedResultSelectedOptions = st.multiselect(
        "Select the task results you need to analyze.",
        resultSelectOptions,
        # label_visibility="hidden",
        default=resultSelectOptions,
    )
    selectedResult = []
    for option in selectedResultSelectedOptions:
        result = results[resultSelectOptions.index(option)].results
        selectedResult += result

    return selectedResult
