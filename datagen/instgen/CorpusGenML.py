class MovieLensCorpus:

    Preference = {
        'iPreference' : [
            """
            User's historical interactions: {interaction}. 
            Based on these movie titles, use your knowledge to generate a description of the user's implicit preferences, 
            such as their favorite genres, themes, or notable patterns.{constraint}
            """,

            """
            The user has browsed the following movies in chronological order: {interaction}. 
            Based on this browsing history, use your understanding of these movies to generate a description of 
            the user's implicit preferences, including their likely favorite genres, themes, or types of movies.{constraint}
            """,

            """
            Recently, the user has browsed the following movies: {interaction}. 
            Based on this recent activity,  apply your knowledge of these movie to generate a description of the user's current movie preferences, 
            focusing on genres, themes, or other noticeable patterns.{constraint}
            """
        ],

        'ePreference' : [
            """
            Analyze the user's recent viewing history: {interaction}. 
            From these interactions, use your knowledge of these movies to infer the user's implicit preferences, 
            such as preferred genres, sub-genres, or specific types of storylines.{constraint}
            """,

            """
            The user has shown a strong interest in the following movies: {interaction}. 
            Using this data, infer their explicit preferences, such as particular themes, moods, 
            or types of narratives they actively seek.{constraint}
            """,

            """
            Consider the user's engagement with the following movies: {interaction}. 
            Based on these patterns, determine their explicit preferences, such as favorite directors, 
            frequent actors, or recurring motifs that they seem to appreciate.{constraint}
            """
        ],

        'nPreference' : [
            """
            The user has recently browsed a variety of different movie genres: {interaction}. 
            Based on this diverse viewing pattern, describe the user's novelty preferences, 
            such as their openness to exploring new genres or trying unexpected movie types.{constraint}
            """,

            """
            Given the user's browsing history: {interaction}, identify any novelty preferences they may have, 
            such as a willingness to explore genres outside their usual interest or a desire for unique 
            and unconventional film experiences.{constraint}
            """,

            """
            The user has moved from browsing typical genres to less common ones: {interaction}. 
            Describe the user's novelty preferences, focusing on their interest in discovering diverse genres 
            or unique cinematic styles.{constraint}
            """
        ],



    }



    #################################################################################

    
    Intention = {

        'vIntention' : [
            """
            The user has recently watched the following movies: {interaction}. 
            Reflect on this history to infer a general type or mood of movies they might be interested in next, 
            without narrowing down to a specific genre or characteristic.{constraint}
            """,

            """
            Given the user's recent viewing history: {interaction}, 
            suggest a broad intention for what they may want to watch next, focusing on an overall style or feeling 
            rather than pinpointing a particular movie or specific genre.{constraint}
            """,

            """
            Based on these movies: {interaction}, generate an open-ended intention that represents a general mood 
            or broad category the user could be leaning towards, even if their specific preferences aren't clear.{constraint}
            """
        ],


        'sIntention' : [
            """
            The user has watched these movies: {interaction}. Use this data to determine a specific movie intention they might have, 
            such as seeking a particular genre, a specific plot, or a film with certain defining characteristics.{constraint}
            """,

            """
            Based on the user's recent movie list: {interaction}, 
            infer a clearly defined intention for the next type of film they may want to watch, focusing on particular elements like genre, theme, or distinctive features.{constraint}
            """,

            """
            Considering the user's viewing pattern: {interaction}, determine a specific intention about the next movie 
            they are likely to watch, including precise details about the genre, mood, or main elements they are interested in.{constraint}
            """
        ],


        'eIntention' : [
            """
            The user has recently watched these movies: {interaction}. Based on this history, 
            suggest an exploratory intention where the user might want to explore genres or types of movies 
            they haven’t typically watched.{constraint}
            """,

            """
            Given the user's movie-watching history of: {interaction}, 
            infer an exploratory intention indicating their curiosity to explore new and different genres, 
            styles, or narrative types that they might not have considered before.{constraint}
            """,

            """
            Using the following interaction data: {interaction}, generate an exploratory intention for the user, 
            where they express interest in trying out new genres, themes, or movie types that differ from their usual choices.{constraint}
            """
        ]
    }

    ###############################################################################

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
