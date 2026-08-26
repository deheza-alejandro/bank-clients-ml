export default {
  plugins: [
    'prefixIds', // Evita que los IDs de un gráfico afecten a otro en el mismo README
    'removeMetadata',     // Elimina bloques <metadata> (XMP, Dublin Core, etc.)
    'removeComments',     // Elimina comentarios XML con fechas de creación
  ],
};