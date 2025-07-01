use super::prompt_request::PromptRequest;
use crate::{
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder, Document,
        Message, Prompt, PromptError,
    },
    streaming::{StreamingChat, StreamingCompletion, StreamingCompletionResponse, StreamingPrompt},
    tool::ToolSet,
    vector_store::VectorStoreError,
};
use futures::{stream, StreamExt, TryStreamExt};
use std::collections::HashMap;

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// # Example
/// ```
/// use rig::{completion::Prompt, providers::openai};
///
/// let openai = openai::Client::from_env();
///
/// let comedian_agent = openai
///     .agent("gpt-4o")
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!")
///     .await
///     .expect("Failed to prompt the agent");
/// ```
pub struct Agent<M: CompletionModel> {
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    pub model: M,
    /// System prompt
    pub preamble: String,
    /// Context documents always available to the agent
    pub static_context: Vec<Document>,
    /// Tools that are always available to the agent (identified by their name)
    pub static_tools: Vec<String>,
    /// Temperature of the model
    pub temperature: Option<f64>,
    /// Maximum number of tokens for the completion
    pub max_tokens: Option<u64>,
    /// Additional parameters to be passed to the model
    pub additional_params: Option<serde_json::Value>,
    /// List of vector store, with the sample number
    pub dynamic_context: Vec<(usize, Box<dyn crate::vector_store::VectorStoreIndexDyn>)>,
    /// Dynamic tools
    pub dynamic_tools: Vec<(usize, Box<dyn crate::vector_store::VectorStoreIndexDyn>)>,
    /// Actual tool implementations
    pub tools: ToolSet,
    /// Whether to include reasoning content in the main content (for reasoning models like deepseek-reasoner)
    pub include_reason_in_content: bool,
    /// Tag to wrap reasoning content when including it in main content
    pub include_reason_in_content_tag: String,
}

impl<M: CompletionModel> Completion<M> for Agent<M> {
    async fn completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let prompt = prompt.into();

        // Find the latest message in the chat history that contains RAG text
        let rag_text = prompt.rag_text();
        let rag_text = rag_text.or_else(|| {
            chat_history
                .iter()
                .rev()
                .find_map(|message| message.rag_text())
        });

        // Merge reasoning configuration with existing additional_params
        let reasoning_params = serde_json::json!({
            "include_reason_in_content": self.include_reason_in_content,
            "include_reason_in_content_tag": self.include_reason_in_content_tag
        });

        let merged_params = if let Some(existing_params) = &self.additional_params {
            crate::json_utils::merge(reasoning_params, existing_params.clone())
        } else {
            reasoning_params
        };

        let completion_request = self
            .model
            .completion_request(prompt)
            .preamble(self.preamble.clone())
            .messages(chat_history)
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params(merged_params)
            .documents(self.static_context.clone());

        // If the agent has RAG text, we need to fetch the dynamic context and tools
        let agent = match &rag_text {
            Some(text) => {
                let dynamic_context = stream::iter(self.dynamic_context.iter())
                    .then(|(num_sample, index)| async {
                        Ok::<_, VectorStoreError>(
                            index
                                .top_n(text, *num_sample)
                                .await?
                                .into_iter()
                                .map(|(_, id, doc)| {
                                    // Pretty print the document if possible for better readability
                                    let text = serde_json::to_string_pretty(&doc)
                                        .unwrap_or_else(|_| doc.to_string());

                                    Document {
                                        id,
                                        text,
                                        additional_props: HashMap::new(),
                                    }
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .try_fold(vec![], |mut acc, docs| async {
                        acc.extend(docs);
                        Ok(acc)
                    })
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

                let dynamic_tools = stream::iter(self.dynamic_tools.iter())
                    .then(|(num_sample, index)| async {
                        Ok::<_, VectorStoreError>(
                            index
                                .top_n_ids(text, *num_sample)
                                .await?
                                .into_iter()
                                .map(|(_, id)| id)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .try_fold(vec![], |mut acc, docs| async {
                        for doc in docs {
                            if let Some(tool) = self.tools.get(&doc) {
                                acc.push(tool.definition(text.into()).await)
                            } else {
                                tracing::warn!("Tool implementation not found in toolset: {}", doc);
                            }
                        }
                        Ok(acc)
                    })
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

                let static_tools = stream::iter(self.static_tools.iter())
                    .filter_map(|toolname| async move {
                        if let Some(tool) = self.tools.get(toolname) {
                            // TODO: tool definitions should likely take an `Option<String>`
                            Some(tool.definition(text.into()).await)
                        } else {
                            tracing::warn!(
                                "Tool implementation not found in toolset: {}",
                                toolname
                            );
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .await;

                completion_request
                    .documents(dynamic_context)
                    .tools([static_tools.clone(), dynamic_tools].concat())
            }
            None => {
                let static_tools = stream::iter(self.static_tools.iter())
                    .filter_map(|toolname| async move {
                        if let Some(tool) = self.tools.get(toolname) {
                            // TODO: tool definitions should likely take an `Option<String>`
                            Some(tool.definition("".into()).await)
                        } else {
                            tracing::warn!(
                                "Tool implementation not found in toolset: {}",
                                toolname
                            );
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .await;

                completion_request.tools(static_tools)
            }
        };

        Ok(agent)
    }
}

// Here, we need to ensure that usage of `.prompt` on agent uses these redefinitions on the opaque
//  `Prompt` trait so that when `.prompt` is used at the call-site, it'll use the more specific
//  `PromptRequest` implementation for `Agent`, making the builder's usage fluent.
//
// References:
//  - https://github.com/rust-lang/rust/issues/121718 (refining_impl_trait)

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Prompt for Agent<M> {
    fn prompt(&self, prompt: impl Into<Message> + Send) -> PromptRequest<M> {
        PromptRequest::new(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Prompt for &Agent<M> {
    fn prompt(&self, prompt: impl Into<Message> + Send) -> PromptRequest<M> {
        PromptRequest::new(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Chat for Agent<M> {
    async fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        let mut cloned_history = chat_history.clone();
        PromptRequest::new(self, prompt)
            .with_history(&mut cloned_history)
            .await
    }
}

impl<M: CompletionModel> StreamingCompletion<M> for Agent<M> {
    async fn stream_completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        // Reuse the existing completion implementation to build the request
        // This ensures streaming and non-streaming use the same request building logic
        self.completion(prompt, chat_history).await
    }
}

impl<M: CompletionModel> StreamingPrompt<M::StreamingResponse> for Agent<M> {
    async fn stream_prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError> {
        self.stream_chat(prompt, vec![]).await
    }
}

impl<M: CompletionModel> StreamingChat<M::StreamingResponse> for Agent<M> {
    async fn stream_chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError> {
        self.stream_completion(prompt, chat_history)
            .await?
            .stream()
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        completion::{CompletionModel, CompletionRequest, CompletionResponse},
        streaming::StreamingCompletionResponse,
        OneOrMany,
    };
    use serde_json::json;
    // Mock completion model for testing
    #[derive(Clone)]
    struct MockCompletionModel {
        response: String,
    }

    impl MockCompletionModel {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[derive(Clone)]
    struct MockStreamingResponse;

    impl CompletionModel for MockCompletionModel {
        type Response = String;
        type StreamingResponse = MockStreamingResponse;

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            Ok(CompletionResponse {
                choice: OneOrMany::one(crate::completion::AssistantContent::text(&self.response)),
                raw_response: self.response.clone(),
            })
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            todo!()
        }
    }

    #[test]
    fn test_agent_include_reason_in_content_default() {
        let model = MockCompletionModel::new("Test response");
        let agent = crate::agent::AgentBuilder::new(model)
            .preamble("Test preamble")
            .build();

        // Default should include reasoning in content (for backward compatibility)
        assert!(agent.include_reason_in_content); 
        assert_eq!(agent.include_reason_in_content_tag, "think");
    }

    #[test]
    fn test_agent_include_reason_in_content_disabled() {
        let model = MockCompletionModel::new("Test response");
        let agent = crate::agent::AgentBuilder::new(model)
            .preamble("Test preamble")
            .include_reason_in_content(false)
            .build();

        assert!(!agent.include_reason_in_content);
        assert_eq!(agent.include_reason_in_content_tag, "think");
    }

    #[test]
    fn test_agent_custom_reason_tag() {
        let model = MockCompletionModel::new("Test response");
        let agent = crate::agent::AgentBuilder::new(model)
            .preamble("Test preamble")
            .include_reason_in_content_tag("reasoning")
            .build();

        assert!(agent.include_reason_in_content);
        assert_eq!(agent.include_reason_in_content_tag, "reasoning");
    }

    #[test]
    fn test_agent_reasoning_config_in_additional_params() {
        let model = MockCompletionModel::new("Test response");
        let agent = crate::agent::AgentBuilder::new(model)
            .preamble("Test preamble")
            .include_reason_in_content(false)
            .include_reason_in_content_tag("custom")
            .additional_params(json!({"custom_param": "value"}))
            .build();

        // Test that the configuration is properly set
        assert!(!agent.include_reason_in_content);
        assert_eq!(agent.include_reason_in_content_tag, "custom");
        
        // The additional_params should contain the custom parameter
        let params = agent.additional_params.as_ref().unwrap();
        assert_eq!(params["custom_param"], "value");
    }

    #[tokio::test]
    async fn test_agent_completion_includes_reasoning_config() {
        let model = MockCompletionModel::new("Test response");
        let agent = crate::agent::AgentBuilder::new(model)
            .preamble("Test preamble")
            .include_reason_in_content(true)
            .include_reason_in_content_tag("thought")
            .build();

        let completion_builder = agent
            .completion("Test prompt", vec![])
            .await
            .expect("Should create completion builder");

        let request = completion_builder.build();
        
        // Check that additional_params contains reasoning configuration
        let params = request.additional_params.as_ref().unwrap();
        assert_eq!(params["include_reason_in_content"], true);
        assert_eq!(params["include_reason_in_content_tag"], "thought");
    }

    #[tokio::test]
    async fn test_agent_completion_merges_reasoning_with_existing_params() {
        let model = MockCompletionModel::new("Test response");
        let agent = crate::agent::AgentBuilder::new(model)
            .preamble("Test preamble")
            .include_reason_in_content(false)
            .include_reason_in_content_tag("analysis")
            .additional_params(json!({
                "temperature": 0.8,
                "custom_field": "custom_value"
            }))
            .build();

        let completion_builder = agent
            .completion("Test prompt", vec![])
            .await
            .expect("Should create completion builder");

        let request = completion_builder.build();
        
        // Check that additional_params contains both original and reasoning configuration
        let params = request.additional_params.as_ref().unwrap();
        assert_eq!(params["include_reason_in_content"], false);
        assert_eq!(params["include_reason_in_content_tag"], "analysis");
        assert_eq!(params["temperature"], 0.8);
        assert_eq!(params["custom_field"], "custom_value");
    }

    // Integration tests that require real DeepSeek API
    // Note: Run with DEEPSEEK_API_KEY set and --ignored flag
    mod integration_tests {
        use super::*;
        use crate::{
            client::{CompletionClient, ProviderClient}, 
            providers::deepseek
        };

        #[tokio::test]
        #[ignore] // Run with: cargo test -- --ignored
        async fn test_deepseek_agent_with_reasoning_included() {
            // This test requires DEEPSEEK_API_KEY environment variable
            let client = deepseek::Client::from_env();
            let agent = client
                .agent(deepseek::DEEPSEEK_REASONER)
                .preamble("你是一个数学老师，请解释你的思考过程。")
                .include_reason_in_content(true)
                .include_reason_in_content_tag("思考")
                .build();

            let response = agent
                .prompt("计算 2 + 2")
                .await
                .expect("Should get response from DeepSeek");

            // Response should contain both reasoning and answer in one text block
            assert!(response.contains("<思考>"));
            assert!(response.contains("</思考>"));
            println!("Response with reasoning included: {}", response);
        }

        #[tokio::test]
        #[ignore] // Run with: cargo test -- --ignored
        async fn test_deepseek_agent_with_reasoning_separated() {
            // This test requires DEEPSEEK_API_KEY environment variable
            let client = deepseek::Client::from_env();
            let agent = client
                .agent(deepseek::DEEPSEEK_REASONER)
                .preamble("你是一个数学老师，请解释你的思考过程。")
                .include_reason_in_content(false)
                .include_reason_in_content_tag("分析")
                .build();

            let response = agent
                .prompt("计算 3 + 3")
                .await
                .expect("Should get response from DeepSeek");

            // In separated mode, the response format might be different
            // This depends on how the streaming/completion response is handled
            println!("Response with reasoning separated: {}", response);
        }
    }
}
