import streamlit as st
def Feature_description(feature):
    if feature == 'CRIM':
        st.sidebar.write('Per capita crime rate by town')
    elif feature == 'INDUS':
        st.sidebar.write("Proportion of non-retail business acres per town")
    elif feature == 'NOX':
        st.sidebar.write("Nitric oxides concentration (parts per 10 million)")
    elif feature == 'RM':
        st.sidebar.write("Average number of rooms per dwelling")
    elif feature == 'AGE':
        st.sidebar.write("Proportion of owner-occupied units built prior to 1940")
    elif feature == 'DIS':
        st.sidebar.write("Weighted distances to ﬁve Boston employment centers")
    elif feature == 'TAX':
        st.sidebar.write("Proportion of non-retail business acres per town")
    elif feature == 'PTRATIO':
        st.sidebar.write("Pupil-teacher ratio by town ")
    else:
        if feature == 'LSTAT':
            st.sidebar.write("% lower status of the population")
            
            
def load_data():
    data = pd.read_csv(r'C:\Users\MASTER\Boston\boston.csv')
    return data