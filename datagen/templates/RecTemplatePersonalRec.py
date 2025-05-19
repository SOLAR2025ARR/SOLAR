class RecTemplatePersonalRec:

    T1 = [
        
        # PN1 IE T1
        """
        {generation} The user likes to explore new types of products and has recently shown interest in items that differ from their usual preferences. 
        The user is looking to try new domains or product types. Based on the user's historical behavior and intention :{historical_interactions}, 
        generate a product recommendation that aligns with the user's novelty preference : {novelty_preference}.
        """,

        # PN1, (P2), IE, T1 
        """
        {generation} The user is interested in exploring new types of products while maintaining certain explicit preferences: {explicit_preference}.  
        Given the user's exploratory intention ({exploratory_intention}) to try something new and different,  
        please generate a product recommendation that aligns with both the user's explicit preferences and their desire for exploration.
        """
    ]

    T2 = [
        # PN2 IE T2 - next
        """
        {direct} The user enjoys receiving surprising recommendations and wants to try items that do not match their usual preferences. 
        Based on the user's exploratory intention:{exploratory_intention} and combine the user's historical action : {historical_interactions}, 
        select the item most likely to offer a pleasant surprise from the following candidates: {candidate_items}
        """,

        # PN1 I1 T2 - yes/no
        """
        {matching} The user is interested in new types of products that do not match their usual preferences:{explicit_preference} but their needs are still unclear. 
        Please determine whether the following item matches the user's vague exploratory intention and answer "Yes" or "No": {candidate_item}
        """,

        # PN2 I0 T2 - next
        """
        {direct} The user has no specific intention but enjoys receiving surprising recommendations. 
        Based on this, select the item most likely to provide a pleasant surprise from the following candidates: {candidate_items}
        """,

        # P1, (P2), IE, T2 - next 
        """
        {direct} The user has shown certain implicit preferences based on their historical interactions: {historical_interactions}. 
        Additionally, the user has explicitly expressed a desire to explore new product types or categories: {novelty_preference}. 
        Given the following candidate items, please select the one that best matches both the user's preference: {candidate_items}
        """,

        # PN1, (P2), I1/I2, T2 - next 
        """
        {direct} The user likes to explore new types of products while also having certain explicit preferences: {explicit_preference}.  
        The user's intention is specific :{specific_intention}.  Based on this combination of preferences and intentions, 
        please select the best option that aligns with the user's needs from the following candidate items: {candidate_items}  
        """,

        # PN2, (P1), IE, T2 - next
        """
        {direct} The user enjoys unexpected recommendations and has implicit preferences derived from their historical interactions: {historical_interactions}.  
        With an exploratory intention ({exploratory_intention}) to discover new interests or products,  
        please select an item from the following candidates: {candidate_items} that is likely to surprise the user.
        """,

        # PN2, (P1), I1/I2, T2 - yes/no
        """
        {matching} The user enjoys being surprised and has shown implicit preferences based on their historical interactions: {historical_interactions}.  
        The user's current intention may be vague as following : {vague_intention} .  
        Based on this information, evaluate the following candidate item: {candidate_item}  
        to determine if it would be a suitable recommendation for the user, please answer "Yes" or "No" for the fitness of candidate.
        """
    ]