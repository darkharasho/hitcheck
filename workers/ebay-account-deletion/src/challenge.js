/**
 * eBay's account-deletion challenge hash.
 *
 * The three strings are concatenated in this exact order and SHA-256'd to
 * lowercase hex. `endpoint` must match what is registered in the eBay
 * developer portal byte for byte -- a trailing-slash or scheme difference
 * produces a well-formed hash that eBay rejects, which is the usual reason
 * verification fails.
 *
 * Kept separate from the request handler so it can be tested without a
 * Workers runtime: `crypto.subtle` is a global in both Workers and Node.
 */
export async function challengeResponse(challengeCode, verificationToken, endpoint) {
  const data = new TextEncoder().encode(challengeCode + verificationToken + endpoint)
  const digest = await crypto.subtle.digest('SHA-256', data)
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('')
}
