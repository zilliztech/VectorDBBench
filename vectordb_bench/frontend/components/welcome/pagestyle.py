def pagestyle():
    html_content = """
    <style>
    .grid-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);  
        grid-template-rows: repeat(3, auto);   
        gap: 20px;
        padding: 20px 0;
    }

    .title-row {
    grid-column: 1 / 4;  
    text-align: left;  
    margin: 20px 0;    
    }

    .title-row h2 {
        font-size: 35px;     
        color: #333;       
        font-weight: 600;   
    }

    .last-row {
    grid-column: 1 / 7;  
    display: grid;
    grid-template-columns: repeat(6, 1fr);  
    gap: 40px;
    }

    .last-row > :nth-child(1) {
        grid-column: 2 / 4;
    }

    .last-row > :nth-child(2) {
        grid-column: 4 / 6;
            gap: 40px;
        }
    .section-card {
        width: 100%;
        height: 350px;
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
        overflow: hidden;
        cursor: pointer;
        display: flex;
        flex-direction: column;
    }
    .section-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        z-index: 100;
    }
    .section-image {
        width: 100%;
        height: 185px;
        object-fit: cover;
        object-position: 0 0;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #262730;
    }
    .section-description {
        font-size: 14px;
        color: #555;
        height: 80px;
        overflow-y: auto;
        margin-bottom: 10px;
    }
    .scroll-container {
        width: 100%;
        overflow-x: auto;
        white-space: nowrap;
        margin-top: auto;
        padding: 10px 0;
        border-top: 1px solid #eee;
    }
    .scroll-content {
        display: inline-block;
        white-space: nowrap;
        padding: 0 10px;
    }
    .scroll-item {
        display: inline-block;
        width: 50px;
        height: 30px;
        margin-right: 10px;
        background-color: #ddd;
        border-radius: 5px;
        text-align: center;
        line-height: 30px;
    }
    </style>

    <div class="grid-container">
    """
    return html_content
