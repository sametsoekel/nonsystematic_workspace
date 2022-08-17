suppressPackageStartupMessages({
    library(data.table)
    library(tidyverse)
    library(tidymodels)
    library(caret)
    library(keras)
    library(catboost)
    library(tensorflow)
    library(collapse)
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
            #data[,(i) := lapply(.SD,function(x) (x*(factors[['max']] - factors[['min']]) + factors[['min']])),.SDcols = i]
        }else{
            #data[,(i) := lapply(.SD,function(x) (x - factors[['min']]) / (factors[['max']] - factors[['min']])),.SDcols = i]
            #data[,(i) := lapply(.SD,function(x) fifelse(is.na(x),-1,x)),.SDcols = i]
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
    #df_dummied <- recipe(df) %>% step_dummy(all_nominal()) %>% prep %>% bake(new_data = NULL)
    df_dummied <- recipe(df) %>% prep %>% step_string2factor(all_nominal()) %>% prep %>% bake(new_data = NULL)
    scaled_data[['data']] <- df_dummied
    scaled_data
}
                          
                          
                          
make_frame <- function(df,test = FALSE,train_frame = NULL,config){
    dt <- copy(df)
    setDT(dt)
    if(!test){
        target_ <- dt[[config$label]]
        
        dt[,(config$label) := NULL]
    }
    sparsed <- sparse_data(dt,sparse_percentage = config$sparse_percentage)
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
    
    dummied[['data']] <- dummied[['data']]# %>% as.matrix
    base::message('All done !')
    dummied
}
                          
create_folds <- function(train_data,config){
    set.seed(config$seed)
    caret::createFolds(train_data$label,k = config$folds)
}
                          
fold_threshold <- function(y_probs,y_test){
    thresholds <- data.frame()
    
    for(j in seq(from = 0.1,to = 0.9,by = 0.01)){
        obs <- y_test
        prd <- +(y_probs >= j)
        if(sum(prd) == 0){
            next
        }
        f1_clc <- f_meas_vec(truth = factor(obs,levels = 0:1),estimate = factor(prd,levels = 0:1),event_level = 'second')
        fold_row <- data.table(threshold = j,f1 = f1_clc)
        thresholds <- rbindlist(list(thresholds,fold_row))
    }
    
    best_threshold <- thresholds %>% filter(f1 == max(f1)) %>% select(threshold) %>% pull %>% .[1]
    error <- thresholds %>% filter(f1 == max(f1)) %>% select(f1) %>% pull %>% .[1]
    
    return(list(threshold = best_threshold,score = error))
}
                          
                          
cross_validate <- function(train_data,target_data,folds,config,label_encoder,label_back_encoder){
    
    set.seed(config$seed)
    
    folds_ <- 1:length(folds)
    
    errors <- c()
    
    preds <- list()
    models <- list()
    feature_importances <- data.table()
    
    prm <- list(learning_rate = config$learning_rate,iterations = config$iter,
                loss_function = config$objective,auto_class_weights = config$auto_class_weights,
                verbose = 0,custom_loss = config$loss_fun,eval_metric = config$eval_metric,
                use_best_model = config$use_best,task_type = config$task,devices = config$device,
                early_stopping_rounds = config$early_stopping_rounds)
    #print(prm)
    IRdisplay::display(config$slicer_1)
    for(i in folds_){
        
        train_indices <- setdiff(folds_,i) 
        test_indices <- i
        
        train_x <- train_data$data[unlist(folds[train_indices]),] 
        test_x <- train_data$data[unlist(folds[test_indices]),]
        
        train_y <- train_data$label[unlist(folds[train_indices])] %>% label_encoder 
        test_y <- train_data$label[unlist(folds[test_indices])] %>% label_encoder 
        
        train_frame <- catboost::catboost.load_pool(data = train_x,label = train_y)
        test_frame <- catboost::catboost.load_pool(data = test_x,label = test_y)
        
        start_it <- Sys.time()
        
        cat_model <- catboost::catboost.train(learn_pool = train_frame,test_pool = test_frame,params = prm)
        
        prediction_to_threshold <- catboost.predict(cat_model,test_frame,prediction_type='Probability')
        
        thresholded <- fold_threshold(y_probs = prediction_to_threshold,y_test = test_y)
        
        msgformat_score <- 'Fold %s F1 : %s'
        msgformat_threshold <- 'Best threshold for Fold %s : %s'
        
        msg_score <- sprintf(msgformat_score,i,round(thresholded$score,4))
        msg_threshold <- sprintf(msgformat_threshold,i,thresholded$threshold)
        
        prediction <- catboost.predict(cat_model,catboost.load_pool(target_data$data),prediction_type='Probability')
        
        prediction_class <- +(prediction >= thresholded$threshold) %>% label_back_encoder
        
        preds[[i]] <- prediction_class
        models[[i]] <- cat_model
        errors[i] <- thresholded$score
        
        fi_row <- cat_model %>% .$feature_importances %>% t %>% as.data.table
        
        feature_importances <- rbindlist(list(feature_importances,fi_row))
        
        finished_it <- Sys.time()
        
        process <- round(as.numeric(difftime(finished_it,start_it,units = 'min')),4)
        
        msgformat_process <- 'Elapsed time for Fold %s : %s minutes'
        
        msg_process <- sprintf(msgformat_process,i,process)
        
        IRdisplay::display(msg_score)
        IRdisplay::display(msg_threshold)
        
        IRdisplay::display(config$slicer_2)
        
        IRdisplay::display(msg_process)
        
        IRdisplay::display(config$slicer_1)
    }
    msgformat2 <- 'CV Mean F1 : %s'
    IRdisplay::display(sprintf(msgformat2,round(mean(errors,na.rm = T),4)))
    
    feature_importance_plot <- feature_importances %>%
    summarise_all(mean) %>%
    t %>%
    as.data.table(keep.rownames = T) %>%
    ggplot(aes(rn,V1))+
    geom_bar(stat = 'identity')+
    xlab('Variable')+
    ylab('Importance')+
    ggtitle('Variable Importance Plot')
    
    return(list(models = models,preds = preds,fi = feature_importance_plot))
    
    
}