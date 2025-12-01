from functools import partial
import streamlit as st
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.custom.displayCustomCase import (
    displayCustomCase,
)
from vectordb_bench.frontend.components.custom.displayCustomStreamingCase import (
    displayCustomStreamingCase,
)
from vectordb_bench.frontend.components.custom.displaypPrams import displayParams
from vectordb_bench.frontend.components.custom.getCustomConfig import (
    CustomCaseConfig,
    CustomStreamingCaseConfig,
    generate_custom_case,
    generate_custom_streaming_case,
    get_custom_configs,
    get_custom_streaming_configs,
    save_custom_configs,
    save_all_custom_configs,
)
from vectordb_bench.frontend.components.custom.initStyle import initStyle
from vectordb_bench.frontend.config.styles import FAVICON, PAGE_TITLE


class CustomCaseManager:
    customCaseItems: list[CustomCaseConfig]

    def __init__(self):
        self.customCaseItems = get_custom_configs()

    def addCase(self):
        new_custom_case = generate_custom_case()
        new_custom_case.dataset_config.name = f"{new_custom_case.dataset_config.name} {len(self.customCaseItems)}"
        self.customCaseItems += [new_custom_case]
        self.save()

    def deleteCase(self, idx: int):
        self.customCaseItems.pop(idx)
        self.save()

    def save(self):
        # Save performance configs along with existing streaming configs
        streaming_configs = get_custom_streaming_configs()
        save_all_custom_configs(self.customCaseItems, streaming_configs)


class StreamingCaseManager:
    streamingCaseItems: list[CustomStreamingCaseConfig]

    def __init__(self):
        self.streamingCaseItems = get_custom_streaming_configs()

    def addCase(self):
        new_streaming_case = generate_custom_streaming_case()
        new_streaming_case.dataset_config.name = (
            f"{new_streaming_case.dataset_config.name} {len(self.streamingCaseItems)}"
        )
        self.streamingCaseItems += [new_streaming_case]
        self.save()

    def deleteCase(self, idx: int):
        self.streamingCaseItems.pop(idx)
        self.save()

    def save(self):
        # Save streaming configs along with existing performance configs
        performance_configs = get_custom_configs()
        save_all_custom_configs(performance_configs, self.streamingCaseItems)


def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=FAVICON,
        # layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    # init style
    initStyle(st)

    # navigate
    NavToPages(st)

    st.title("Custom Dataset")
    displayParams(st)

    # Performance Test Datasets Section
    st.subheader("Performance Test Datasets")
    st.markdown("These datasets are used for search performance tests.")

    customCaseManager = CustomCaseManager()

    for idx, customCase in enumerate(customCaseManager.customCaseItems):
        expander = st.expander(customCase.dataset_config.name, expanded=True)
        key = f"custom_case_{idx}"
        displayCustomCase(customCase, expander, key=key)

        columns = expander.columns(8)
        columns[0].button(
            "Save",
            key=f"{key}_",
            type="secondary",
            on_click=lambda: customCaseManager.save(),
        )
        columns[1].button(
            ":red[Delete]",
            key=f"{key}_delete",
            type="secondary",
            # B023
            on_click=partial(lambda idx: customCaseManager.deleteCase(idx), idx=idx),
        )

    st.button(
        "+ New Dataset",
        key="add_custom_configs",
        type="primary",
        on_click=lambda: customCaseManager.addCase(),
    )

    st.divider()

    # Streaming Test Datasets Section
    st.subheader("Streaming Test Datasets")
    st.markdown("These datasets are used for streaming performance tests (insertion + search).")

    streamingCaseManager = StreamingCaseManager()

    for idx, streamingCase in enumerate(streamingCaseManager.streamingCaseItems):
        expander = st.expander(streamingCase.dataset_config.name, expanded=True)
        key = f"streaming_case_{idx}"
        displayCustomStreamingCase(streamingCase, expander, key=key)

        columns = expander.columns(8)
        columns[0].button(
            "Save",
            key=f"{key}_save",
            type="secondary",
            on_click=lambda: streamingCaseManager.save(),
        )
        columns[1].button(
            ":red[Delete]",
            key=f"{key}_delete",
            type="secondary",
            on_click=partial(lambda idx: streamingCaseManager.deleteCase(idx), idx=idx),
        )

    st.button(
        "+ New Streaming Dataset",
        key="add_streaming_config",
        type="primary",
        on_click=lambda: streamingCaseManager.addCase(),
    )


if __name__ == "__main__":
    main()
