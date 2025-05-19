class BooksCorpuswoReview:

    Preference = {

        'iPreference': [
            """
            Analyze the user's reading history: {interaction}.  
            From these data points, determine the user's explicit preferences, such as the genres, themes, 
            or specific book characteristics they explicitly praise. {constraint}
            """,

            """
            Considering the user's reading history of these books: {interaction}, 
            generate a description of the user's explicit preferences, focusing on any recurring genres, themes, 
            or patterns evident in their comments. {constraint}
            """,

            """
            Based on the user's recent engagement with the following books: {interaction},  
            identify their explicit preferences by analyzing the sentiments and focus of their engagement, 
            such as preferred genres, themes, or author styles they frequently mention or praise. {constraint}
            """
        ],


        'ePreference': [
            """
            Analyze the user's reading history: {interaction}.  
            From these data points, determine the user's explicit preferences, such as the genres, themes, 
            or specific book characteristics they explicitly praise. {constraint}
            """,

            """
            The user has shown a clear interest in certain books: {interaction}.  
            Using this data, infer their explicit preferences, such as favorite themes, plot types, 
            or narrative styles they often highlight. {constraint}
            """,

            """
            Consider the user's engagement with these books: {interaction}.  
            Based on this, identify explicit preferences, such as preferred authors, frequent genres, 
            or writing styles that the user frequently praises or critiques. {constraint}
            """
        ],


        'nPreference': [
            """
            The user has recently reviewed a variety of different genres or unconventional books: {interaction}. 
            Describe the user's novelty preferences, such as their openness to experimenting with new genres or exploring unique literary styles, 
            based on the diversity of their reading history. {constraint}
            """,

            """
            Given the user's diverse reading history: {interaction}, 
            identify any novelty preferences they may have, such as a tendency to seek out unique literary experiences or genres 
            that are outside their usual interests. {constraint}
            """,

            """
            The user has moved from reading typical genres to exploring less common ones: {interaction}. 
            Describe the user's novelty preferences, 
            focusing on their interest in discovering new genres or unconventional narrative approaches. {constraint}
            """
        ],
    }

    constraint = [
        """
        Based on this reading history, provide a brief and creative summary of the user's book preferences. 
        Feel free to explore different angles, such as their possible taste, mood, or literary interests, 
        without focusing on specific genres or categories. *Your response must be a single, concise paragraph.** {{SingleParagraph}}
        """
    ]

    Intention = {

        'vIntention': [
            """
            The user has recently read the following books: {interaction}. 
            Reflect on this history to infer a general type or mood of books they might be interested in next, 
            without narrowing down to a specific genre or characteristic. {constraint}
            """,

            """
            Given the user's recent reading history: {interaction}, 
            suggest a broad intention for what they may want to read next, focusing on an overall style or feeling 
            rather than pinpointing a particular book or specific genre. {constraint}
            """,

            """
            Based on these books: {interaction}, 
            generate an open-ended intention that represents a general mood or broad category the user could be leaning towards, 
            even if their specific preferences aren't clear. {constraint}
            """
        ],

        'sIntention': [
            """
            The user has read these books: {interaction}. 
            Use this data to determine a specific book intention they might have, 
            such as seeking a particular genre, a specific plot, or a book with certain defining characteristics. {constraint}
            """,

            """
            Based on the user's recent book list: {interaction}, 
            infer a clearly defined intention for the next type of book they may want to read, 
            focusing on particular elements like genre, theme, or distinctive features. {constraint}
            """,

            """
            Considering the user's reading pattern: {interaction}, 
            determine a specific intention about the next book they are likely to read, 
            including precise details about the genre, mood, or main elements they are interested in. {constraint}
            """
        ],

        'eIntention': [
            """
            The user has recently read these books: {interaction}. 
            Based on this history, suggest an exploratory intention where the user might want to explore genres or types of books 
            they haven’t typically read. {constraint}
            """,

            """
            Given the user's book-reading history of: {interaction}, 
            infer an exploratory intention indicating their curiosity to explore new and different genres, 
            styles, or narrative types that they might not have considered before. {constraint}
            """,

            """
            Using the following interaction data: {interaction}, 
            generate an exploratory intention for the user, where they express interest in trying out new genres, themes, 
            or book types that differ from their usual choices. {constraint}
            """
        ]
    }

    ##############################################################################################

    OutputFormat = [

        """
        """,

        """
        Let’s think step by step. {{CoT}}.  
        """,

        """
        Consider any differences or similarities. {{Consider}}
        """,

        """
        Envision scenarios inspired by the user's history. {{Creative Context}}
        """,

        """
        Speculate on the user's preferences. {{Open Hypothesis}}
        """,

        """
        Explore potential interests or hidden patterns. {{Exploratory Analysis}}.
        """
    ]

    constraint = [
        """
        Write a single, concise paragraph (2-3 sentences) that focuses on the most notable aspects, using diverse perspectives while avoiding unnecessary details.
        """
        ]
