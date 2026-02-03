def explainPrams(st):
    st.markdown("## descriptions")
    st.markdown("### 1. Overview")
    st.markdown(
        """
- **VectorDBBench(VDBBench)** is an open-source benchmarking tool designed specifically for vector databases. Its main features include:
    - (1) An easy-to-use **web UI** for configuration of tests and visual analysis of results.
    - (2) A comprehensive set of **standards for testing and metric collection**.
    - (3) Support for **various scenarios**, including additional support for **Filter** and **Streaming** based on standard tests.
- VDBBench embraces open-source and welcome contributions of code and test result submissions. The testing process and extended scenarios of VDBBench, as well as the intention behind our design will be introduced as follows.
"""
    )
    st.markdown("### 2. Dataset")
    st.markdown(
        """
- We provide two embedding datasets:
    - (1)*[Cohere 768dim](https://huggingface.co/datasets/Cohere/wikipedia-22-12)*, generated using the **Cohere** model based on the Wikipedia corpus. 
    - (2)*[Cohere 1024dim](https://huggingface.co/datasets/Cohere/beir-embed-english-v3)*, generated using the **Cohere** embed-english-v3.0 model based on the bioasq corpus.
    - (3)*OpenAI 1536dim*, generated using the **OpenAI** model based on the [C4 corpus](https://huggingface.co/datasets/legacy-datasets/c4).
"""
    )
    st.markdown("### 3. Standard Test")
    st.markdown(
        """
The test is actually divided into 3 sub-processes
- **3.1 Test Part 1 - Load (Insert + Optimize)**
    - (1) Use a single process to perform serial inserts until all data is inserted, and record the time taken as **insert_duration**.
    - (2) For most vector databases, index construction requires additional time to optimize to achieve an optimal state, and record the time taken as **optimize_duration**.
    - (3) **Load_duration (insert_duration + optimize_duration)** can be understood as the time from the start of insertion until the database is ready to query.
        - load_duration can serve as a reference for the insert capability of a vector database to some extent. However, it should be noted that some vector databases may perform better under **concurrent insert operations**.
- **3.2 Test Part 2 - Serial Search Test**
    - (1) Use a single process to perform serial searches, record the results and time taken for each search, and calculate **recall** and **latency**.
    - (2) **Recall**: For vector databases, most searches are approximately nearest neighbor(ANN) searches rather than perfectly accurate results. In production environments, commonly targeted recall rates are 0.9 or 0.95.
        - Note that there is a **trade-off** between **accuracy** and **search performance**. By adjusting parameters, it is possible to sacrifice some accuracy in exchange for better performance. We recommend comparing performance while ensuring that the recall rates remain reasonably close.
    - (3) **Latency**:**p99** rather than average. **latency_p99** focuses on **the slowest 1% of requests**. In many high-demand applications, ensuring that most user requests stay within acceptable latency limits is critical, whereas **latency_avg** can be skewed by faster requests.
        - **serial_latency** can serve as a reference for a database's search capability to some extent. However, serial_latency is significantly affected by network conditions. We recommend running the test client and database server within the same local network.
- **3.3 Test Part 3 - Concurrent Search Test**
    - (1) Create multiple processes, each perform serial searches independently to test the database's **maximum throughput(max-qps)**.
    - (2) Since different databases may reach peak throughput under different conditions, we conduct multiple test rounds. The number of processes **starts at 1 by default and gradually increases up to 80**, with each test group running for **30 seconds**.
        - Detailed latency and QPS metrics at different concurrency levels can be viewed on the <a href="concurrent" target="_self" style="text-decoration: none;">*concurrent*</a> page.
        - The highest recorded QPS value from these tests will be selected as the final max-qps.
""",
        unsafe_allow_html=True,
    )
    st.markdown("### 4. Filter Search Test")
    st.markdown(
        """
- Compared to the Standard Test, the **Filter Search** introduces additional scalar constraints (e.g. **color == red**) during the Search Test. Different **filter_ratios** present varying levels of challenge to the VectorDB's search performance.
- We provide an additional **string column** containing 10 labels with different distribution ratios (50%,20%,10%,5%,2%,1%,0.5%,0.2%,0.1%). For each label, we conduct both a **Serial Test** and a **Concurrency Test** to observe the VectorDB's performance in terms of **QPS, latency, and recall** under different filtering conditions.
"""
    )
    st.markdown("### 5. Streaming Search Test")
    st.markdown(
        """
Different from Standard's load and search separation, Streaming Search Test primarily focuses on **search performance during the insertion process**. 
Different **base dataset sizes** and varying **insertion rates** set distinct challenges to the VectorDB's search capabilities.
VDBBench will send insert requests at a **fixed rate**, maintaining consistent insertion pressure. The search test consists of three steps as follows:
- 1.**Streaming Search** 
    - Users can configure **multiple search stages**. When the inserted data volume reaches a specified stage, a **Serial Test** and a **Concurrent Test** will be conducted, recording qps, latency, and recall performance.
- 2.**Streaming Final Search**
    - After all of the data is inserted, a Serial Test and a Concurrent Test are immediately performed, recording qps, latency, and recall performance. 
        - Note: at this time, the insertion pressure drops to zero since data insertion is complete.
- 3.**Optimized Search (Optional)**
    - Users can optionally perform an additional optimization step followed by a Serial Test and a Concurrent Test, recording qps, latency, and recall performance. This step **compares performance in Streaming section with the theoretically optimal performance**.
"""
    )
