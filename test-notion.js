const DB_ID = "353d33df1d5880159492f219933f1ea0";
const TOKEN = process.env.NOTION_TOKEN;

if (!TOKEN) {
  console.error("❌ Error: Please provide the NOTION_TOKEN environment variable.");
  console.log("Usage: NOTION_TOKEN=your_secret_token node test-notion.js");
  process.exit(1);
}

async function testNotion() {
  console.log("Testing Notion API connection...");
  
  try {
    const response = await fetch("https://api.notion.com/v1/pages", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${TOKEN}`,
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        parent: { database_id: DB_ID },
        properties: {
          Name: {
            title: [{ text: { content: "Test Email: Urgent Action Required!" } }]
          }
        },
        children: [
          {
            object: "block",
            heading_2: { rich_text: [{ text: { content: "Email Details" } }] }
          },
          {
            object: "block",
            paragraph: { rich_text: [{ text: { content: `From: testing-bot@example.com` } }] }
          },
          {
            object: "block",
            paragraph: { rich_text: [{ text: { content: `Summary: This is a test email sent from your local machine to verify the Notion connection.` } }] }
          },
          {
            object: "block",
            paragraph: { rich_text: [{ text: { content: `Why it's important: We are verifying that the Notion token and Database ID are working correctly before deploying to Cloudflare.` } }] }
          }
        ]
      })
    });

    if (response.ok) {
      const data = await response.json();
      console.log("✅ Success! The test page was successfully created in your Notion database.");
      console.log("🔗 You can view it here:", data.url);
    } else {
      const error = await response.text();
      console.error("❌ Failed to create page in Notion. API Response:");
      console.error(error);
    }
  } catch (err) {
    console.error("❌ Network error occurred:", err.message);
  }
}

testNotion();
