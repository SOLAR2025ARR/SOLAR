class RecTemplate:  

    # T0 (rerank)
    # T1 (generative recommendation)
    # T2 (match yes / no OR multiple choice)

    T0 = [
        
        # P1 (P2) I0 T0 
        """
        {reranking} The behavioral sequence of the user is shown below: {historical_interactions}, 
        which can be used to infer the  user’s preferences {explicit_preference}. 
        Then please rerank the items to better align with the user's preferences by comparing the candidates 
        and their similarities to the user's preferences. The candidates are: {candidate_items}.
        """,

        # P1 (P2) I0 T0
        """
        {reranking} You have observed that the user has clicked on the following items: {historical_interactions}, indicating his
        personal tastes: {explicit_preference}. Based on this information, please reranke items from the following candidates 
        that you think would be suitable for the user: {candidate_items}
        """,

        # P1 (P2) I0 T0
        """
        {reranking} You have some information about this user, which is shown below: {explicit_preference}, 
        the user’s historical interactions: {historical_interactions} Based on this information, 
        please recommend the reranking order of items for the user, which should match the user’s preference, 
        from the following candidates: {candidate_items}
        """,

        # P1 (P2) I0 T0
        """ 
        {reranking} You have obtained the user’s historical interaction list, which is as follows:{historical_interactions}. 
        Based on this history, you can infer the user’s preferences {explicit_preference}. 
        Now, you need to give a recommened order of items to recommend to the user. 
        Please rerank the following candidates: {candidate_items}
        """
    ]

    T1 = [
        # P1 I0 T1
        """
        {generation} Using the user’s historical interactions as input data,predict the next product that the user is most likely to interact with.
        The historical interactions are provided as follows: {historical_interactions}. 
        """,

        # 
        """
        {generation} Given the user's interaction history: {historical_interactions}, what is the optimal product to suggest next?
        """,

        """
        {generation} Given the sequence of the user's past interactions: {historical_interactions}, what is the most suitable product to recommend next?
        """,

        """
        {generation} Considering the user's interaction pattern: {historical_interactions}, suggest the next likely product they would engage with."
        """,

        """
        {generation} Given the historical context of user interactions: {historical_interactions}, what is the optimal next product recommendation?
        """,

        """
        {generation} Based on the user's historical engagement data: {historical_interactions}, provide the next product recommendation.
        """,

        """
        {generation} Based on the user's past interaction data: {historical_interactions}, suggest the most relevant product for their next interaction.
        """,

        """
        {generation} Using the provided interaction history: {historical_interactions}, determine the most likely product the user would engage with next.
        """,

        """
        {generation} Analyzing the user's engagement history: {historical_interactions}, generate the best possible product recommendation for their next interaction.
        """,

        """
        {generation} Given the pattern of past interactions: {historical_interactions}, provide a recommendation for the next product that the user is likely to find appealing.
        """,

        """
        {generation} Taking into account the user's prior interactions: {historical_interactions}, recommend the next best product they would likely be interested in.
        """,

        # P1 P2 I0 T1
        """
        {generation} Recommend the next potential product to a user based on his profile and past interaction. You have access to the
        user’s profile information, including his preference: explicit preference and past interactions: 
        {historical_interactions}. Now you need to determine what product would be recommended to him.
        """,
        
        # P2 I0 T1
        """ 
        {generation} You are a recommender system, and are good at recommending products to a user based on his preferences. 
        Given the user’s preferences: {explicit_preference}, please recommend products that are consistent with those preferences
        """,

        # P2 I0 T1
        """
        {generation} As we know, a user’s behavior is driven by his preferences, which determine what they are likely to buy next. 
        Your task is to predict what products a user will purchase next, based on his preferences. 
        Given the user’s preferences as follows: {explicit_preference}, please make your prediction.
        """

        # P1 (P2) I0 T1 - 1
        """
        {generation} Given the following historical interaction of the user: {historical_interactions}. 
        You can infer the user’s preference. {explicit_preference}. Please predict next possible item for the user.
        """,

        # P1 (P2) I0 T1 - 2
        """
        {generation} To make a recommendation for this user, we need to analyze their historical interactions, which are shown below:
        {historical_interactions}. As we know, historical interactions an reflect the user’s preferences.
        Based on this user’s preferences {explicit_preference}, please recommend an item that you think would be suitable for them.
        """,
        
        # P1 P2 I0 T1 - ICL
        """
        {generation} Recommend the next potential product to a user based on his profile and past interactions. You have access to
        the user’s profile information, including his preference: explicit preference and past interactions:
        {historical_interactions}. For example, if the user recently interacted with {recent_item}, you might consider similar products.
        Now, based on this approach, determine what product would be recommended to him next.
        """,

        # P1 (P2) I0 T1 - 1 - ICL
        """
        {generation} Given the following historical interaction of the user: {historical_interactions}. 
        For instance, if the user recently interacted with {recent_item}, you might predict similar items as the next possible choice.
        Based on this user's preferences {explicit_preference}, please predict the next possible item for the user.
        """

        # P1 I0 T1 - ICL
        """
        {generation} Imagine the user recently interacted with {recent_item}. Using this example, 
        and given the user's historical interactions as input data: {historical_interactions}, 
        predict the next product that the user is most likely to interact with.
        """

        # P1 (P2) I0 T1 - 2 - ICL
        """
        {generation} Consider that the user recently interacted with {recent_item}. 
        Given this example, and based on the user’s historical interactions: {historical_interactions}, 
        analyze the user’s preferences. Taking into account these preferences: {explicit_preference}, 
        please recommend an item that would be suitable for them.
        """
    ]

    T2 = [
        # P1 (P2) I0 T2 - yes/no
        """
        {direct} The user has previously purchased the following items: {historical_interactions}. 
        This information indicates their personalized preferences {explicit_preference}. 
        Based on this information, is it likely that the user will interact with
        {candidate_item} next?
        """

        # P1 (P2) I0 T2 - yes/no
        """
        {direct} Based on the user’s historical interaction list, which
        is provided as follows: {historical_interactions} , you can infer
        the user’s personalized preference {explicit_preference}. And
        your task is to use this information to predict whether the user will
        click on {candidate_item} next.
        """

        # P1 (P2) I0 T2 - next
        """
        {direct} Please recommend an item to the user based on the following information about the user: {historical_interactions},
        the user’s historical interaction, which is as follows: {explicit_preference} 
        Try to select one item from the following candidates that is consistent with the user’s preference: {candidate_items}.
        """
    ]

    Complex = [
        
        # P0 P2/I1/I2 T1
        """
        {generation} Suppose you are a search engine, now the user
        search that {explicit_preference_vague_intention_specific_intention}, 
        can you generate the item to respond to user’s query?
        """

        # P0 P2/I1/I2 T1
        """
        {generation} If a user asks a question like:{explicit_preference_vague_intention_specific_intention}
        Please generate a related suitable item to help him.
        """
    ]