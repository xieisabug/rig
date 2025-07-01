use rig::{completion::Prompt, providers::deepseek, client::{ProviderClient, CompletionClient}};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化 DeepSeek 客户端
    let client = deepseek::Client::from_env();

    println!("=== DeepSeek 思考内容配置示例 ===\n");

    // 示例 1: 默认行为 - 思考内容会包含在正式回答中
    println!("=== 示例 1: 默认行为 (思考内容包含在回答中) ===");
    let agent_default = client
        .agent(deepseek::DEEPSEEK_REASONER)
        .preamble("你是一个数学老师，请解释你的思考过程。")
        .build();

    let response1 = agent_default
        .prompt("计算 25 * 34，并解释计算过程")
        .await?;
    println!("默认行为响应:\n{}\n", response1);

    // 示例 2: 配置不包含思考内容在正式回答中
    println!("=== 示例 2: 分离思考内容 ===");
    let agent_separate = client
        .agent(deepseek::DEEPSEEK_REASONER)
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
        .agent(deepseek::DEEPSEEK_REASONER)
        .preamble("你是一个数学老师，请解释你的思考过程。")
        .include_reason_in_content(true)
        .include_reason_in_content_tag("reasoning")
        .build();

    let response3 = agent_custom_tag
        .prompt("计算 25 * 34，并解释计算过程")
        .await?;
    println!("自定义标签响应:\n{}\n", response3);

    println!("=== 示例完成 ===");
    println!("注意: 要使用实际的 API，请设置 DEEPSEEK_API_KEY 环境变量");

    Ok(())
} 