class MoviesTVCorpuswoReview:

    Preference = {

        'iPreference': [
            """
            Analyze the user's viewing history: {interaction}.  
            From these data points, determine the user's explicit preferences, such as the genres, themes, 
            or specific movie characteristics they explicitly praise. {constraint}
            """,

            """
            Considering the user's viewing history of these movies/TV shows: {interaction},  
            generate a description of the user's implicit preferences, focusing on any recurring genres, themes, 
            or patterns evident in their engagement. {constraint}
            """,

            """
            Based on the user's recent engagement with the following movies/TV shows: {interaction},  
            identify their implicit preferences by analyzing the sentiments and focus of their engagement, 
            such as preferred genres, themes, or character types. {constraint}
            """
        ],

        'ePreference' : [
            """
            Analyze the user's viewing history: {interaction}.  
            From these data points, determine the user's explicit preferences, such as the genres, themes, 
            or specific movie characteristics they explicitly praise. {constraint}
            """,

            """
            The user has shown a clear interest in certain movies/TV shows: {interaction}.  
            Using this data, infer their explicit preferences, such as favorite themes, plot types, 
            or emotional tones they often highlight. {constraint}
            """,

            """
            Consider the user's engagement with these movies/TV shows: {interaction}.  
            Based on this, identify explicit preferences, such as preferred directors, frequent actors, 
            or narrative styles that the user frequently praises or critiques. {constraint}
            """
        ],

        'nPreference' : [
            """
            The user has recently reviewed a variety of different genres or unconventional movies/TV shows: {interaction}. 
            Describe the user's novelty preferences, such as their openness to experimenting with new genres or exploring unique cinematic styles, 
            based on the diversity of their viewing history. {constraint}
            """,

            """
            Given the user's diverse viewing history: {interaction}, 
            identify any novelty preferences they may have, such as a tendency to seek out unique film experiences or genres 
            that are outside their usual interests. {constraint}
            """,

            """
            The user has moved from watching typical genres to exploring less common ones: {interaction}.  
            Describe the user's novelty preferences, 
            focusing on their interest in discovering new genres or unconventional storytelling approaches. {constraint}
            """
        ]

    }

    

    Intention = {

            'vIntention': [
            """
            The user has recently watched the following movies: {interaction}. 
            Reflect on this history to infer a general type or mood of movies they might be interested in next, 
            without narrowing down to a specific genre or characteristic. {constraint}
            """,

            """
            Given the user's recent viewing history: {interaction}, 
            suggest a broad intention for what they may want to watch next, focusing on an overall style or feeling 
            rather than pinpointing a particular movie or specific genre. {constraint}
            """,

            """
            Based on these movies: {interaction}, 
            generate an open-ended intention that represents a general mood or broad category the user could be leaning towards, 
            even if their specific preferences aren't clear. {constraint}
            """
        ],

        'sIntention': [
            """
            The user has watched these movies: {interaction}. 
            Use this data to determine a specific movie intention they might have, 
            such as seeking a particular genre, a specific plot, or a film with certain defining characteristics. {constraint}
            """,

            """
            Based on the user's recent movie list: {interaction}, 
            infer a clearly defined intention for the next type of film they may want to watch, 
            focusing on particular elements like genre, theme, or distinctive features. {constraint}
            """,

            """
            Considering the user's viewing pattern: {interaction}, 
            determine a specific intention about the next movie they are likely to watch, 
            including precise details about the genre, mood, or main elements they are interested in. {constraint}
            """
        ],

        'eIntention': [
            """
            The user has recently watched these movies: {interaction}.  
            Based on this history, suggest an exploratory intention where the user might want to explore genres or types of movies 
            they haven’t typically watched. {constraint}
            """,

            """
            Given the user's movie-watching history of: {interaction}, 
            infer an exploratory intention indicating their curiosity to explore new and different genres, 
            styles, or narrative types that they might not have considered before. {constraint}
            """,

            """
            Using the following interaction data: {interaction}, 
            generate an exploratory intention for the user, where they express interest in trying out new genres, themes, 
            or movie types that differ from their usual choices. {constraint}
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
