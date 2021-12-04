def gender_from_name(df, name_col):
    def get_gender(s):
        m = re.search("(Sir|Lady|Mrs?|Ms|Miss)\.", s)
        if m:
            if m.group(1) == "Mr" or m.group(1) == "Sir":
                return "Male"
            else:
                return "Female"
        else:
            # print(s)
            return np.nan

    return df[name_col].apply(lambda s: get_gender(s))
