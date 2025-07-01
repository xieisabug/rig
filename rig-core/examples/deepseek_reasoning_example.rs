use rig::providers::deepseek::{Message, CompletionResponse, DEEPSEEK_REASONER};
use rig::prelude::*;
use rig::{completion::{Completion, Prompt}, providers};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DeepSeek Reasoning Content Example ===\n");

    // Example 1: Basic reasoning content deserialization
    println!("1. Basic Reasoning Content Parsing:");
    let reasoning_response_json = r#"
    {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The square root of 16 is 4.",
                    "reasoning_content": "To find the square root of 16, I need to find a number that when multiplied by itself equals 16. Let me think: 1×1=1, 2×2=4, 3×3=9, 4×4=16. So the answer is 4."
                },
                "logprobs": null,
                "finish_reason": "stop"
            }
        ]
    }
    "#;

    let response: CompletionResponse = serde_json::from_str(reasoning_response_json).unwrap();
    let choice = response.choices.first().unwrap();
    
    if let Message::Assistant { content, reasoning_content, .. } = &choice.message {
        println!("  Content: {}", content);
        if let Some(reasoning) = reasoning_content {
            println!("  Reasoning: {}", reasoning);
        }
    }
    println!();

    // Example 2: DeepSeek R1 style reasoning with <think> tags
    println!("2. DeepSeek R1 Style Reasoning:");
    let r1_response_json = r#"
    {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I recommend using a divide-and-conquer approach to solve this problem efficiently.",
                    "reasoning_content": "<think>\nThe user is asking about algorithm optimization. This looks like a classic computer science problem that could benefit from different algorithmic approaches.\n\nLet me consider the options:\n1. Brute force - Simple but inefficient for large inputs\n2. Dynamic programming - Good for problems with overlapping subproblems\n3. Divide and conquer - Excellent for problems that can be broken down recursively\n4. Greedy algorithms - Fast but doesn't always guarantee optimal solutions\n\nGiven the context, divide-and-conquer seems most appropriate as it can significantly reduce the time complexity.\n</think>"
                },
                "logprobs": null,
                "finish_reason": "stop"
            }
        ]
    }
    "#;

    let r1_response: CompletionResponse = serde_json::from_str(r1_response_json).unwrap();
    let r1_choice = r1_response.choices.first().unwrap();
    
    if let Message::Assistant { content, reasoning_content, .. } = &r1_choice.message {
        println!("  Final Answer: {}", content);
        if let Some(reasoning) = reasoning_content {
            println!("  Internal Reasoning Process:");
            println!("  {}", reasoning);
        }
    }
    println!();

    // Example 3: Serialization (without reasoning_content)
    println!("3. Serialization Examples:");
    let simple_message = Message::Assistant {
        content: "Hello, world!".to_string(),
        reasoning_content: None,
        name: None,
        tool_calls: vec![],
    };
    
    let serialized_simple = serde_json::to_string(&simple_message).unwrap();
    println!("  Simple message (no reasoning): {}", serialized_simple);
    
    // Example 4: Serialization (with reasoning_content)
    let reasoning_message = Message::Assistant {
        content: "The answer is 42.".to_string(),
        reasoning_content: Some("I need to think about this carefully. The question of life, the universe, and everything has been answered in Douglas Adams' work as 42.".to_string()),
        name: None,
        tool_calls: vec![],
    };
    
    let serialized_reasoning = serde_json::to_string(&reasoning_message).unwrap();
    println!("  Message with reasoning: {}", serialized_reasoning);
    println!();

    // Example 5: Using Agent with DeepSeek Reasoner to access reasoning content
    println!("4. Agent with DeepSeek Reasoner Example:");
    
    // First, let's show the mock response analysis to understand the structure
    println!("  Mock Response Analysis (showing expected structure):");
    let mock_math_response = r#"
    {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "To solve 2x + 5 = 15:\n\n1. Subtract 5 from both sides: 2x = 10\n2. Divide by 2: x = 5\n\nTherefore, x = 5.",
                    "reasoning_content": "<think>\nI need to solve the equation 2x + 5 = 15 step by step.\n\nStep 1: I want to isolate the term with x, so I'll subtract 5 from both sides.\n2x + 5 - 5 = 15 - 5\n2x = 10\n\nStep 2: Now I need to get x by itself, so I'll divide both sides by 2.\n2x ÷ 2 = 10 ÷ 2\nx = 5\n\nLet me verify: 2(5) + 5 = 10 + 5 = 15 ✓\n\nSo the answer is x = 5.\n</think>"
                },
                "logprobs": null,
                "finish_reason": "stop"
            }
        ]
    }
    "#;
    
    let math_response: CompletionResponse = serde_json::from_str(mock_math_response).unwrap();
    if let Some(choice) = math_response.choices.first() {
        if let Message::Assistant { content, reasoning_content, .. } = &choice.message {
            println!("    Final Answer: {}", content);
            if let Some(reasoning) = reasoning_content {
                println!("    Internal Thinking:");
                println!("    {}", reasoning);
            }
        }
    }
    
    println!();
    
    // Now show how to use Agent to get this kind of response
    println!("  Live Agent Demo (requires DEEPSEEK_API_KEY):");
    if std::env::var("DEEPSEEK_API_KEY").is_ok() {
        let client = providers::deepseek::Client::from_env();
        
        // Create an agent using the DeepSeek Reasoner model (this is the correct way!)
        let agent = client
            .agent(DEEPSEEK_REASONER)
            .preamble("You are a helpful math tutor. Show your reasoning step by step.")
            .temperature(0.1)
            .build();
        
        // Use agent.completion() to get full response with reasoning
        let prompt = "Explain step by step how to solve 2x + 5 = 15";
        match agent.completion(prompt, vec![]).await {
            Ok(request_builder) => {
                match request_builder.send().await {
                    Ok(response) => {
                        println!("  Question: {}", prompt);
                        
                        // The response.choice contains the final assistant content including reasoning
                        match response.choice.into_iter().next() {
                            Some(rig::completion::AssistantContent::Text(text)) => {
                                println!("  Agent Answer (includes reasoning): {}", text.text);
                            },
                            _ => println!("  No text response received"),
                        }
                        
                        // The raw_response contains the original DeepSeek response with separate reasoning
                        if let Some(choice) = response.raw_response.choices.first() {
                            if let Message::Assistant { reasoning_content, .. } = &choice.message {
                                if let Some(reasoning) = reasoning_content {
                                    println!("  Raw Reasoning Content (also included above): {}", reasoning);
                                } else {
                                    println!("  No reasoning content in this response");
                                }
                            }
                        }
                    },
                    Err(e) => println!("  Error sending request: {}", e),
                }
            },
            Err(e) => println!("  Error building completion: {}", e),
        }
        
        // Also demonstrate the simpler agent.prompt() method (includes reasoning in text)
        println!("\n  Comparison - Using agent.prompt() (simpler, reasoning included in response):");
        match agent.prompt("What is 3 + 4?").await {
            Ok(simple_answer) => {
                println!("  Simple Answer (includes reasoning): {}", simple_answer);
                println!("  Note: With .prompt(), reasoning content is included in the text response");
            },
            Err(e) => println!("  Error with simple prompt: {}", e),
        }
    } else {
        println!("    → Set DEEPSEEK_API_KEY environment variable to run live agent demo");
        println!("    → The agent response now includes reasoning content in the text output");
        println!("    → Use .completion() to access both combined text and separate reasoning fields");
        println!("    → Use .prompt() for simple text responses (reasoning included in text)");
    }
    
    println!();
    println!("=== Example completed successfully! ===");
    println!("Note: To use the live API, set your DEEPSEEK_API_KEY environment variable");
    
    // 初始化 DeepSeek 客户端
    let client = providers::deepseek::Client::from_env();

    // 示例 1: 默认行为 - 思考内容会包含在正式回答中
    println!("=== 示例 1: 默认行为 ===");
    let agent_default = client
        .agent(DEEPSEEK_REASONER)
        .preamble("你是一个数学老师，请解释你的思考过程。")
        .build();

    let response1 = agent_default
        .prompt("计算 25 * 34，并解释计算过程")
        .await?;
    println!("默认行为响应:\n{}\n", response1);

    // 示例 2: 配置不包含思考内容在正式回答中
    println!("=== 示例 2: 分离思考内容 ===");
    let agent_separate = client
        .agent(DEEPSEEK_REASONER)
        .preamble("你是一个数学老师，请解释你的思考过程。")
        .include_reason_in_content(false)
        .build();

    let response2 = agent_separate
        .prompt("计算 25 * 34，并解释计算过程")
        .await?;
    println!("分离思考内容响应:\n{}\n", response2);

    // 示例 3: 自定义思考内容标签
    println!("=== 示例 3: 自定义思考标签 ===");
    let agent_custom_tag = client
        .agent(DEEPSEEK_REASONER)
        .preamble("你是一个数学老师，请解释你的思考过程。")
        .include_reason_in_content(true)
        .include_reason_in_content_tag("reasoning")
        .build();

    let response3 = agent_custom_tag
        .prompt("计算 25 * 34，并解释计算过程")
        .await?;
    println!("自定义标签响应:\n{}\n", response3);

    Ok(())
}