class RecTemplatePersonalSearch:

    T1 = [
        # P1 P2 T1
        """
        {generation} You are a search engine. Here is the historical interaction of a user: {historical_interactions}. 
        And his personalized preferences are as follows: {explicit_preference}. 
        Your task is to generate a new product that are consistent with the user’s preference.
        """,

        # P1 I1 T1
        """ 
        {generation} The user has interacted with a list of items, which are as follows: {historical_interactions}. 
        Based on these interacted items, the user current intent are as follows {vague_intention},
        and your task is to generate products that match the user’s currentintent.
        """

        # P1 I1 T1
        """
        {generation} As a search engine, you are assisting a user who is searching for the query: {specific_intention}. 
        Your task is to recommend products that match the user’s query and also align
        with their preferences based on their historical interactions, 
        which are reflected in the following: {historical_interactions}
        """
    ]


    T2 = [
        # P1(P2), P2/I1/I2, T2 - next
        """
        {direct} Using the user’s current query: {explicit_preference_vague_intention_specific_intention} 
        and their historical interactions: {historical_interactions} you can estimate the user’s preferences {explicit_preference} 
        Please respond to the user’s query by selecting an item from the following candidates that best matches their preference and query:
        {candidate_items}
        """

        # P1(P2), P2/I1/I2, T2 - next 
        """
        {direct} The user wants to try some products and searches for: {explicit_preference_vague_intention_specific_intention}. 
        In addition, they have previously bought: {historical_interactions} 
        You can estimate their preference by analyzing his historical interactions. {explicit_preference}
        Please recommend one of the candidate items below that best matches their search query and preferences: {candidate_items}
        """

        # P1(P2), P2/I1/I2, T2 - next
        """
        {direct} A user enjoys shopping very much and has purchased a lot of goods. They are: {historical_interactions}.
        His historical interactions can reflect his personalized preference. {explicit_preference}. 
        Now he wants to try some new items, such as: ’{explicit_preference_vague_intention_specific_intention}’ 
        The recommender system recommends the following candidate items based on his needs and preferences: {candidate_items} 
        Please select the item that best meets the user’s needs and preferences from among the candidate items.
        """

        # P1(P2), P2/I1/I2, T2 -  next
        """
        {direct} Based on the user’s historical interactions with the following items: {historical_interactions} 
        You can infer his preference by analyzing the historical interactions. {explicit_preference} 
        Now the user wants to try a new item and searches for: “{explicit_preference_vague_intention_specific_intention}” 
        Please select a suitable item from the following candidates that matches his preference and search intent: {candidate_items}
        """
    ]