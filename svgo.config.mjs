export default {
  plugins: [
    'prefixIds', // This prevents IDs in one chart from affecting another in the same README.
    'removeMetadata',     // Remove <metadata> blocks (XMP, Dublin Core, etc.)
    'removeComments',     // Remove XML comments containing creation dates
  ],
};