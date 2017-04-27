module ParseBabi

export createVocab, parseBabiTask, createVocabJoint, parseBabiTaskJoint

function parseBabiTask(vocab, baseDir, taskID; isTest = false)
	path = baseDir * searchdir(taskID, isTest, baseDir)
	return parseBabiTaskHelper([path], vocab)
end

function createVocab(baseDir, taskID)
	trainPath = baseDir * searchdir(taskID, false, baseDir)
	testPath = baseDir * searchdir(taskID, true, baseDir)

	vocab = Dict{String,Int}()
	createVocabHelper(vocab, trainPath)
	createVocabHelper(vocab, testPath)
	return vocab
end

function createVocabJoint(baseDir)
	vocab = Dict{String,Int}()
	for taskID in 1:20
		trainPath = baseDir * searchdir(taskID, false, baseDir)
		testPath = baseDir * searchdir(taskID, true, baseDir)
		createVocabHelper(vocab, trainPath)
		createVocabHelper(vocab, testPath)
	end
	return vocab
end

function parseBabiTaskJoint(vocab, baseDir)
	paths = Array(String, 20)
	for taskID in 1:20
		paths[taskID] = baseDir * searchdir(taskID, false, baseDir)
	end
	return parseBabiTaskHelper(paths, vocab)
end

function parseBabiTaskHelper(paths, vocab)
	story = zeros(Int, 15, 250, 7000) # (word, sentence, story)
	qstory = zeros(Int, 15, 20000) # (word, question_idx)
	questions = zeros(Int, 3, 20000) # (info, question_idx), info = (answer, sentences_before, story)
	story_idx = 0
	sentence_idx = 0
	question_idx = 0
	max_words = 0
	max_sentences = 0

	for path in paths
		open(path) do f
	    	for line in enumerate(eachline(f))
	    		is_question = false
	    		if contains(line[2], "?")
	    			is_question = true
	    			question_idx = question_idx + 1
	    		else
	    			sentence_idx = sentence_idx + 1
	    		end

	    		arr = split(line[2])
	    		if parse(Int, arr[1]) == 1
	    			story_idx = story_idx + 1
	    			if max_sentences < sentence_idx
	    				max_sentences = sentence_idx
	    			end
	    			sentence_idx = 1
	    		end

	    		for i in 2:length(arr)
	    			word = lowercase(arr[i])
	    			answer_next = false
	    			if word[end] == '.' || word[end] == '?'
	    				answer_next = true
	    				word = word[1:end-1]
	    			end

	    			if (i - 1 > max_words)
	    				max_words = i - 1
	    			end

	    			if is_question
	    				qstory[i - 1, question_idx] = vocab[word]
	    				if answer_next
	    					questions[1, question_idx] = vocab[lowercase(arr[i + 1])]
	    					questions[2, question_idx] = sentence_idx
	    					questions[3, question_idx] = story_idx
	    					break
	    				end
	    			else
	    				story[i - 1, sentence_idx, story_idx] = vocab[word]
	    			end

	      		end
	    	end
		end
	end
	story = story[1:max_words, 1:max_sentences, 1:story_idx]
	qstory = qstory[1:max_words, 1:question_idx]
	questions = questions[:, 1:question_idx]
	return story, qstory, questions
end

function createVocabHelper(vocab, path)
	open(path) do f
    	for line in enumerate(eachline(f))
    		arr = split(line[2])
    		for i in 2:length(arr)
    			word = lowercase(arr[i])
    			if word[end] == '.'
    				word = word[1:end-1]
    				get!(vocab, word, length(vocab)+1)

    			elseif word[end] == '?'
    				word = word[1:end-1]
    				get!(vocab, word, length(vocab)+1)
    				get!(vocab, arr[i + 1], length(vocab)+1)
    				break
    			else
    				get!(vocab, word, length(vocab)+1)
    			end
      		end
    	end
	end
end

function searchdir(taskID, isTest, baseDir)
	key = "qa" * string(taskID) * "_"
	if isTest
		return filter(x->(contains(x, key) && contains(x, "test")), readdir(baseDir))[1]
	else
		return filter(x->(contains(x, key) && contains(x, "train")), readdir(baseDir))[1]
	end
end

end