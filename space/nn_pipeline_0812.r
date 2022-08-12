suppressPackageStartupMessages({
    library(data.table)
    library(tidyverse)
    library(tidymodels)
})

sparse_data <- function(df,sparse_percentage = 0.02){
    
    df_new <- copy(df) 
    setDT(df_new)
    categorical_names <- df_new %>% purrr::discard(is.numeric) %>% colnames

    for(i in categorical_names){
        features <- df_new[,unique(.SD),.SDcols = i] %>% pull
        
        for(j in features){
            
            condition_format <- 'df_new[,length(%s[%s == "%s"])/.N < %s]'
            condition_command <- sprintf(condition_format,i,i,j,sparse_percentage)
        
            condition <- eval(parse(text = condition_command))
            
            if(condition){
                sparse_format <- 'df_new[%s == "%s", %s := "other"]'
                sparse_command <- sprintf(sparse_format,i,j,i)
                eval(parse(text = sparse_command))
            } 
        }
    }
    df_new
}

get_scaling_factors <- function(df){
    setDT(df)
    factors <- lapply(df %>% keep(is.numeric),function(x) list(min = min(x,na.rm = T),max = max(x,na.rm = T)))
    factors
}     
                      
scale_data <- function(df,scaling_factors,reverse = FALSE){
    
    data <- copy(df)
    setDT(data)
    for(i in names(scaling_factors)){
        factors <- scaling_factors[[i]]
        if(reverse){
            data[,(i) := lapply(.SD,function(x) (x*(factors[['max']] - factors[['min']]) + factors[['min']])),.SDcols = i]
        }else{
            data[,(i) := lapply(.SD,function(x) (x - factors[['min']]) / (factors[['max']] - factors[['min']])),.SDcols = i]
            data[,(i) := lapply(.SD,function(x) fifelse(is.na(x),-1,x)),.SDcols = i]
        }  
    }
                                
    catcols <- df %>% purrr::discard(is.numeric) %>% colnames
    out <- list(scaling_factors = scaling_factors,data = data,cat_cols = catcols)
    return(out)
} 
                                
clean_test_set <- function(scaled_data,categorical_cols,distinct_values_on_train){
    df <- copy(scaled_data[['data']])
    
    make_paste <- function(x){
    wrapped <- sapply(x,function(x) paste0('"',x,'"'))
    paste0(wrapped,collapse = ',')
    }       
   
    
    for(i in categorical_cols){
        distincts <- distinct_values_on_train[[i]]
        distincts <- make_paste(distincts)
        command_format <- "df[! %s %%in%% c(%s), %s := 'Missing']"
        command <- sprintf(command_format,i,distincts,i)
        eval(parse(text = command))
    }
                      
    scaled_data[['data']] <- df
                      
    scaled_data
}
                      
get_distincts <- function(scaled_data){
    df <- copy(scaled_data[['data']])
    
    catcols <- scaled_data[['cat_cols']]
    
    distincts <- list()
    for(i in catcols){
        distinct_values <- df[,unique(.SD),.SDcols = i] %>% pull %>% as.character
        distincts[[i]] <- c(distinct_values,'Missing')
        df[,(i) := lapply(.SD,function(x) as.character(x)),.SDcols = i]
        df[,(i) := lapply(.SD,function(x) ifelse(is.na(x),'Missing',x)),.SDcols = i]
        df[,(i) := lapply(.SD,function(x) factor(x,levels = c(distinct_values,'Missing'))),.SDcols = i]
    }
    
    

    scaled_data[['cat_distincts']] <- distincts
    
    scaled_data[['data']] <- df
                          
    scaled_data
}

fetch_test_levels_to_train <- function(scaled_data,categorical_cols,distinct_values_on_train){
    df <- copy(scaled_data[['data']])
    
    for(i in categorical_cols){
        distincts <- distinct_values_on_train[[i]]
        df[,(i) := lapply(.SD,function(x) factor(x,levels = distincts)),.SDcols = i]
    }
   
    scaled_data[['data']] <- df
    scaled_data
}

dummy_data <- function(scaled_data){
    df <- copy(scaled_data[['data']])
    df_dummied <- recipe(df) %>% step_dummy(all_nominal()) %>% prep %>% bake(new_data = NULL)
    scaled_data[['data']] <- df_dummied
    scaled_data
}
                          
                          
                          
make_frame <- function(df,label,test = FALSE,train_frame = NULL,sparse_perc = 0.03){
    dt <- copy(df)
    setDT(dt)
    if(!test){
        target_ <- dt[[label]]
        
        dt[,(label) := NULL]
    }
    sparsed <- sparse_data(dt,sparse_percentage = sparse_perc)
    gc()
    base::message('Data sparsed.')
    
    if(!test){
        scl <- get_scaling_factors(sparsed)
    }else{
        scl <- train_frame[['scaling_factors']]
    }
    gc()
    base::message('Scaling factors calculated.')
    base::message('Missing values labeled.')
    scld <- scale_data(sparsed,scl)
    gc()
    base::message('Data scaled.')
    
    if(test){
        scld <- clean_test_set(scld,train_frame[['cat_cols']],train_frame[['cat_distincts']])
        base::message('Unseen values removed from test set.')
        scld <- fetch_test_levels_to_train(scld,train_frame[['cat_cols']],train_frame[['cat_distincts']])
        base::message('Train & test set nominal levels fetched.')
    }
    
    if(!test){
    distincted <- get_distincts(scld)
    gc()
    base::message('Got distinct values for nominals.')
    }else{
        distincted <- scld
    }
    
    dummied <- dummy_data(distincted)
    gc()
    base::message('Data dummied.')
    
    if(!test){
        dummied[['label_to_keras']] <- target_# %>% label_encode %>% keras::to_categorical()
        dummied[['label']] <- target_ #%>% label_encode
    }
    
    dummied[['data']] <- dummied[['data']] %>% as.matrix
    base::message('All done !')
    dummied
}