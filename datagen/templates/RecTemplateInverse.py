class RecTemplateInverse:


    # intention的太少了


    T0 = [

        # P1 (P2) I0 T0_-1 
        """
        {explicit_preference}The behavioral sequence of the user is shown below: {historical_interactions}. The candidates were provided as: {candidate_items}, 
        and they have been reranked to better align with the user's preferences: {rerank_list}.  
        Based on this information, please infer the user's explicit preferences that likely led to this reranking.
        """,

        # P1 (P2) I0 T0_-1
        """
        {explicit_preference}You have observed that the user has clicked on the following items: {historical_interactions}.  
        The following candidates were presented: {candidate_items}, and they have been reranked in an order deemed suitable for the user: {rerank_list}.  
        Based on this information, please infer the user's explicit preferences that likely led to this reranking.
        """,

        # P1 (P2) I0 T0_-1
        """
        {explicit_preference}You have some information about this user, which is shown below: the user’s historical interactions: {historical_interactions}.  
        The candidates presented were: {candidate_items}, and they have been reranked in the following order: {rerank_list}.  
        Based on this information, please infer the user's explicit preferences that would justify this reranking.
        """,

        # P1 I0 T0_-1
        """
        {implicit_preference}The user has interacted with the following items in the past: {historical_interactions}.  
        The candidates provided were: {candidate_items}, and they have been reranked to better align with the user's interests: {rerank_list}.  
        Based on this information, please infer the user's implicit preferences that likely led to this reranking.
        """,

        # P1 I1 T0_-1
        """
        {implicit_preference}You have observed the user's behavior with these items: {historical_interactions}.  
        The following candidates were presented: {candidate_items}, and they have been reordered as follows: {rerank_list}.  
        Based on this information, infer the user's implicit preferences and any vague intentions that might justify this order.
        """,

        """
        {vague_intention}The user has shown the following historical interactions: {historical_interactions}, and the candidate items were provided as: {candidate_items}.  
        The candidates have been reranked in this order: {rerank_list}.  
        Based on this information, infer the user's vague intention that could explain why this reranking aligns with their preferences.
        """

        # P1 I1 T0_-1
        """
        {vague_intention}The user has shown the following behavioral sequence: {historical_interactions}.  
        The candidates were presented as: {candidate_items} and have been reranked as follows: {rerank_list}.  
        Based on this data, infer the user's vague intention, considering what general type or mood of items the user may be inclined towards, even if it isn't clear.
        """,

        # P1 I2 T0_-1
        """
        {specific_intention}The user's interaction history is as follows: {historical_interactions}.  
        The candidates were ranked in the following order: {rerank_list}, and originally provided as: {candidate_items}.  
        Given this information, infer the user's specific intention that likely led to this ordering, such as a preference for a particular theme, genre, or type of item.
        """,

        # P2 I2 T0_-1
        """
        {specific_intention}Analyzing the user's past behavior: {historical_interactions} and the given candidates: {candidate_items},  
        which have been reordered to: {rerank_list}, please determine the user's specific intention that could explain this preference for certain elements over others.
        """,

        # PN1 I1 T0_-1
        """
        {vague_intention}Given the user's historical interactions: {historical_interactions}, and the provided candidates: {candidate_items} which were reordered as: {rerank_list},  
        please infer the user's vague intention that might have influenced this specific ranking, focusing on broad interests or exploratory tendencies.
        """
    ]

    T1 = [
        # P1 (P2) I0 T1_-1 - 1
        """
        {explicit_preference}Given the following historical interaction of the user: {historical_interactions}.  
        And the next recommended item: {next_item}. Please infer the user’s explicit preferences that would likely lead to this recommendation.
        """,

        # P1 (P2) I0 T1_-1 - 2
        """
        {explicit_preference}To understand this user's preferences, we need to analyze their historical interactions, which are shown below: {historical_interactions}.  
        The next recommended item is: {next_item}.  
        Based on this information, please infer the user's explicit preferences that would have led to this recommendation.
        """,
        
        # PN1 IE T1_-1
        """
        {novelty_preference}Given the user's historical behavior and intention: {historical_interactions}, and the next recommended item: {next_item},  
        please infer the user's exploratory preferences that would justify this recommendation.
        """,

        # P1 I2 T1_-1
        """
        {specific_intention}Given the following historical interactions of the user: {historical_interactions},  
        and the next recommended item: {next_item}. Please infer the specific intention that would likely lead to this recommendation, such as seeking a particular genre, theme, or type of item.
        """,

        # P1 (P2) I2 T1_-1
        """
        {specific_intention}To better understand the user's needs, consider their past interactions: {historical_interactions}.  
        The next recommended item is: {next_item}.  
        Based on this information, infer the user's specific intention that would justify this recommendation, focusing on concrete preferences or desires.
        """
    ]


    T2 = [
        
        # PN2 IE T2_-1 - next
        """
        {exploratory_intention}The user has recently been recommended the following item: {next_item}.  
        Given the user's historical actions: {historical_interactions} and the candidates: {candidate_items},  
        please infer the user's exploratory intention that would justify this surprising recommendation.
        """,

        # PN1 I1 T2_-1 - yes/no
        """
        {exploratory_intention}The user has shown interest in a specific item: {next_item}.  
        Given their historical interactions: {historical_interactions} and the available candidate items: {candidate_items},  
        infer the user's vague exploratory intention that would justify this interest.  
        Please describe the intention in a way that aligns with the selection of this item.
        """,

        # PN2 I0 T2_-1 - next
        """
        {exploratory_intention} The user was recommended the following item: {next_item}.  
        Considering their historical interactions: {historical_interactions} and the set of candidates: {candidate_items},  
        please infer the user's lack of specific intention for surprising recommendations that justify the selection of this item.
        """,

        # P1 (P2) I0 T2 
        """
        {explicit_preference} Based on the user’s historical interaction list, which is provided as follows: {historical_interactions},  
        and given the following candidate items: {candidate_items}, the item that was selected as most likely to be clicked next is: {next_item}.  
        Please infer the user's personalized preference that would justify the selection of this item.
        """,

        # P1 (P2) I0 T2 
        """
        {explicit_preference} Based on the user’s historical interaction list, which is provided as follows: {historical_interactions},  
        and from the following candidates: {candidate_items}, the item selected as most likely to be clicked next is: {next_item}.  
        Using this information, please infer the user's personalized preferences that would justify the selection of this item.
        """,

        # P1 (P2) I0 T2_-1
        """
        {explicit_preference} Please try to infer the preference to the user based on the following information: {historical_interactions}, 
        the user’s historical interaction, which is as follows: {next_item} and the candidate item: {candidate_items}.
        """,

        # PN1 I1 T2_-1
        """
        {vague_intention} The user has received the following recommendation: {next_item}.  
        Given their historical actions: {historical_interactions} and the set of candidates: {candidate_items},  
        please infer the user's vague intention that could justify this recommendation.
        """,

        # P1 (P2) I0 T2_-1
        """
        {implicit_preference} Based on the user's historical interaction list: {historical_interactions},  
        and considering the candidate items: {candidate_items}, the item most likely to be clicked next is: {next_item}.  
        Please infer the user's implicit preferences that would justify the selection of this item.
        """
    ]