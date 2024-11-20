from datetime import time
import os
from pathlib import Path
import hashlib
import random
import base64
from PIL import Image
import json
import shutil
import cssutils
import re
import gzip
import string
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ResourceConfig:

    """Configuration schema for ResourceSplitter"""
    resource_fingerprinting: Dict[str, bool] = None

    def __post_init__(self):
        if self.resource_fingerprinting is None:
            self.resource_fingerprinting = {
                "alternate_resource_loading": True,
                "randomize_image_format": True,
                "split_resources": True
            }

        # Validate required keys
        required_keys = ["alternate_resource_loading",
                         "randomize_image_format", "split_resources"]
        for key in required_keys:
            if key not in self.resource_fingerprinting:
                raise ValueError(f"Missing required configuration key: {key}")


class ResourceSplitter:
    """
    Handles resource fingerprinting and transformation techniques:
    - Randomize file names with build hashes
    - Split/combine resources differently each build
    - Alternate between inline and external resources
    - Add content-hash to asset URLs
    - Randomize image format when equivalent
    - Base64 encode resources randomly
    - Use data URIs with random encodings
    - Add random srcset variations
    - Mix between CSS gradients and images
    """
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS = {
        'image': ['.jpg', '.jpeg', '.png', '.webp'],
        'style': ['.css'],
        'script': ['.js'],
        'vector': ['.svg'],
        'font': ['.ttf', '.woff', '.woff2']
    }
    COMPRESSION_QUALITY = {
        'high': (85, 95),
        'medium': (60, 84),
        'low': (40, 59)
    }

    def __init__(self, base_dir: Path, config: dict):
        """Initialize ResourceSplitter with configuration and logging"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"resource_splitter_{time.strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        try:
            # Validate requirements first
            if not self._check_requirements():
                raise ImportError("Missing required dependencies")

            # Validate configuration
            if not self._validate_config(config):
                raise ValueError("Invalid configuration provided")

            # Validate base directory
            if not isinstance(base_dir, Path):
                base_dir = Path(base_dir)
            
            if not base_dir.exists():
                self.logger.info(f"Creating base directory: {base_dir}")
                base_dir.mkdir(parents=True, exist_ok=True)

            # Initialize configuration
            self.config = ResourceConfig(**config).__dict__
            self.base_dir = base_dir
            
            # Initialize internal state
            self.resource_map = {}
            self.content_hashes = {}
            self.build_id = self._generate_build_id()
            self.dependencies = {}
            self.cache_variations = {}
            self.format_variations = {}
            self.conditional_loads = []
            self.metadata = {}
            self.encoded_variations = {}
            self.chunks = {}
            self.fallbacks = {}
            self.font_variations = {}
            self.compressed_paths = {}

            # Create required subdirectories
            self._init_directory_structure()

            self.logger.info(f"ResourceSplitter initialized with build ID: {self.build_id}")

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def _init_directory_structure(self):
        """Initialize required directory structure"""
        try:
            for dir_name in ['images', 'css', 'js', 'fonts']:
                dir_path = self.base_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {dir_path}")
        except Exception as e:
            self.logger.error(f"Error creating directory structure: {str(e)}")
            raise

    def _check_requirements(self) -> bool:
        """Check if all required libraries are available"""
        required_libs = {
            'PIL': 'Pillow>=9.0.0',
            'cssutils': 'cssutils>=2.7.0'
        }

        missing_libs = []
        for lib, version in required_libs.items():
            try:
                __import__(lib)
            except ImportError:
                missing_libs.append(version)

        if missing_libs:
            self.logger.error(f"Missing required libraries: {', '.join(missing_libs)}")
            return False
        return True

    def _validate_config(self, config: dict) -> bool:
        """Validate configuration dictionary"""
        required_keys = {
            "resource_fingerprinting": {
                "alternate_resource_loading",
                "randomize_image_format",
                "split_resources"
            }
        }

        try:
            if not isinstance(config, dict):
                self.logger.error("Configuration must be a dictionary")
                return False

            if not all(key in config for key in required_keys):
                self.logger.error(f"Missing required top-level keys: {required_keys.keys()}")
                return False

            fingerprinting = config.get("resource_fingerprinting", {})
            if not isinstance(fingerprinting, dict):
                self.logger.error("resource_fingerprinting must be a dictionary")
                return False

            missing_keys = required_keys["resource_fingerprinting"] - set(fingerprinting.keys())
            if missing_keys:
                self.logger.error(f"Missing required fingerprinting keys: {missing_keys}")
                return False

            # Validate value types
            for key, value in fingerprinting.items():
                if not isinstance(value, bool):
                    self.logger.error(f"Configuration value for {key} must be boolean")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating config: {str(e)}")
            return False

    def process_resources(self) -> Dict[str, str]:
        """Main entry point for processing all resources"""
        processed_count = 0
        error_count = 0

        # Track processed files to avoid duplicates
        processed_files = set()

        try:
            # Validate base directory exists
            if not self.base_dir.exists():
                raise FileNotFoundError(
                    f"Base directory {self.base_dir} does not exist")

            # Initial setup
            try:
                self._generate_resource_map_file()
                self.dependencies = self._create_resource_dependencies()
            except Exception as e:
                self.logger.error(f"Error in initial setup: {str(e)}")
                raise

            for file_path in self.base_dir.rglob('*'):
                # Skip directories and already processed files
                if file_path.is_dir() or str(file_path) in processed_files:
                    continue

                try:
                    if self._should_process(file_path):
                        self.logger.info(f"Processing file: {file_path}")

                        # Pre-processing
                        try:
                            timestamp_vars = self._add_timestamp_variations()
                            self._obfuscate_file_metadata(file_path)
                            encoded_variations = self._create_encoding_variations(
                                file_path.read_bytes())
                        except Exception as e:
                            self.logger.warning(
                                f"Pre-processing error for {file_path}: {str(e)}")
                            # Continue with core processing even if pre-processing fails

                        # Core processing
                        transformed_path = self._process_file(file_path)
                        if transformed_path:
                            try:
                                # Process CSS specific
                                if transformed_path.suffix.lower() == '.css':
                                    css_content = transformed_path.read_text(
                                        encoding='utf-8')
                                    updated_css = self._update_css_urls(
                                        css_content)
                                    split_result = self._split_css_file(
                                        updated_css, transformed_path)
                                    if split_result:
                                        transformed_path = split_result

                                # Image/font processing
                                if transformed_path.suffix.lower() in ['.jpg', '.png']:
                                    try:
                                        with Image.open(transformed_path) as img:
                                            self._generate_srcset_variations(
                                                img, transformed_path.suffix[1:])
                                    except Exception as e:
                                        self.logger.error(
                                            f"Error generating image variations: {str(e)}")

                                elif transformed_path.suffix.lower() == '.ttf':
                                    self._generate_font_variations(
                                        transformed_path)

                                # Post-processing
                                integrity_hash = self._generate_resource_integrity(
                                    transformed_path)
                                compressed_path = self._apply_random_compression(
                                    transformed_path)
                                fallbacks = self._create_resource_fallbacks(
                                    transformed_path)
                                chunks = self._apply_resource_chunking(
                                    transformed_path.read_bytes())

                                # Add metadata
                                self.metadata[str(transformed_path)] = {
                                    'integrity': integrity_hash,
                                    'compressed_path': str(compressed_path) if compressed_path else None,
                                    'fallbacks': [str(p) for p in fallbacks] if fallbacks else [],
                                    'chunks': len(chunks) if chunks else 0
                                }

                                # Verification
                                if self._verify_transformations(file_path, transformed_path):
                                    self.resource_map[str(file_path)] = str(
                                        transformed_path)
                                    processed_files.add(str(file_path))
                                    processed_count += 1
                                else:
                                    self.logger.warning(
                                        f"Verification failed for {file_path}")
                                    error_count += 1

                            except Exception as e:
                                self.logger.error(
                                    f"Error processing {file_path}: {str(e)}")
                                error_count += 1
                                continue

                except Exception as e:
                    self.logger.error(
                        f"Fatal error processing {file_path}: {str(e)}")
                    error_count += 1
                    continue

        except Exception as e:
            self.logger.error(
                f"Critical error in resource processing: {str(e)}")
            raise
        finally:
            # Cleanup and final processing
            try:
                self._cleanup_temporary_resources()
                self._update_resource_map_file()  # Update with final state
            except Exception as e:
                self.logger.error(f"Error in cleanup: {str(e)}")

            # Log processing summary
            self.logger.info(
                f"Processing complete. Processed: {processed_count}, Errors: {error_count}")

        return self.resource_map

    def _should_process(self, file_path: Path) -> bool:
        """Determine if file should be processed based on extension and config"""
        try:
            if file_path.stat().st_size > self.MAX_FILE_SIZE:  # Use class constant
                self.logger.warning(f"Skipping {file_path}: File too large")
                return False

            # Skip hidden files
            if file_path.name.startswith('.'):
                return False

            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                self.logger.warning(f"Skipping {file_path}: File not readable")
                return False

            return any(file_path.suffix.lower() in formats
                       for formats in self.SUPPORTED_FORMATS.values())

        except Exception as e:
            self.logger.error(f"Error checking file {file_path}: {str(e)}")
            return False

    def _process_file(self, file_path: Path) -> Optional[Path]:
        """
        Process individual file based on type.
        Returns new path after transformations.
        """
        try:
            if not self._validate_path(file_path):
                self.logger.error(f"Invalid file path: {file_path}")
                return None
            # Validate input file
            if not file_path.exists():
                raise FileNotFoundError(f"File {file_path} does not exist")

            if not file_path.is_file():
                raise ValueError(f"{file_path} is not a regular file")

            # Calculate content hash
            content_hash = self._calculate_content_hash(file_path)
            self.content_hashes[str(file_path)] = content_hash

            # Determine if we should inline this resource
            if self.config['resource_fingerprinting']['alternate_resource_loading']:
                if random.random() < 0.3:  # 30% chance to inline
                    inlined = self._convert_to_data_uri(file_path)
                    if inlined:
                        return None  # Resource is now inline

            # Process based on file type
            suffix = file_path.suffix.lower()
            if suffix in ['.jpg', '.png']:
                return self._process_image(file_path)
            elif suffix == '.css':
                return self._process_css(file_path)
            elif suffix == '.js':
                return self._process_js(file_path)
            elif suffix == '.svg':
                return self._process_svg(file_path)
            else:
                self.logger.warning(f"Unhandled file type: {suffix}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def _calculate_content_hash(self, file_path: Path) -> str:
        """Calculate content hash for resource fingerprinting"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    def _process_image(self, file_path: Path) -> Path:
        """
        Transform image files:
        - Randomize format (jpg/png/webp)
        - Generate multiple resolutions for srcset
        - Optionally convert to base64
        - Add random quality variations
        """
        try:
            with Image.open(file_path) as img:

                # Decide output format
                formats = ['png', 'jpeg', 'webp'] if img.mode == 'RGBA' else [
                    'jpeg', 'webp']
                output_format = random.choice(formats)

                # Generate srcset variations
                if self.config['resource_fingerprinting']['randomize_image_format']:
                    srcset_paths = self._generate_srcset_variations(
                        img, output_format)
                    self.resource_map[f"{str(file_path)}_srcset"] = json.dumps(
                        srcset_paths)

                # Create fingerprinted filename
                new_name = f"{file_path.stem}_{self.build_id}.{output_format}"
                output_path = file_path.parent / new_name

                # Save with random quality
                quality = random.randint(85, 95)
                img.save(output_path, format=output_format, quality=quality)

                return output_path

        except Exception as e:
            self.logger.error(f"Error processing image {file_path}: {str(e)}")
            return file_path

    def _generate_srcset_variations(self, img: Image, format: str) -> List[Dict[str, str]]:
        """Generate multiple image sizes for srcset attribute"""
        widths = [0.5, 0.75, 1, 1.5, 2]  # Multipliers for original size
        variations = []

        for width_mult in widths:
            new_width = int(img.width * width_mult)
            new_height = int(img.height * width_mult)
            resized = img.resize((new_width, new_height), Image.LANCZOS)

            output_name = f"{self.build_id}_{new_width}w.{format}"
            output_path = self.base_dir / "images" / output_name

            resized.save(output_path, format=format, quality=90)
            variations.append({
                "path": str(output_path),
                "width": new_width
            })

        return variations

    def _convert_to_data_uri(self, file_path: Path) -> None:
        """Convert file to base64 data URI"""
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.css': 'text/css',
            '.js': 'application/javascript'
        }

        mime_type = mime_types.get(file_path.suffix.lower())
        if not mime_type:
            return None

        with open(file_path, 'rb') as f:
            data = f.read()
            b64_data = base64.b64encode(data).decode()

        self.resource_map[str(file_path)
                          ] = f"data:{mime_type};base64,{b64_data}"
        return None  # Indicates resource is now inline

    def _process_css(self, file_path: Path) -> Path:
        """
        Transform CSS files:
        - Split into multiple files randomly
        - Add fingerprint to URLs
        - Convert between CSS and inline styles
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            css_content = f.read()

        # Update resource URLs in CSS
        css_content = self._update_css_urls(css_content)

        if self.config['resource_fingerprinting']['split_resources']:
            return self._split_css_file(css_content, file_path)
        else:
            # Create fingerprinted filename
            new_name = f"{file_path.stem}_{self.build_id}.css"
            output_path = file_path.parent / new_name

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(css_content)

            return output_path

    def _update_css_urls(self, css_content: str) -> str:
        """Update URLs in CSS to use fingerprinted versions"""
        def replace_url(match):
            url = match.group(1)
            if url in self.resource_map:
                return f"url({self.resource_map[url]})"
            return match.group(0)

        return re.sub(r'url\((.*?)\)', replace_url, css_content)

    def _split_css_file(self, css_content: str, original_path: Path) -> Path:
        """Split CSS into multiple files randomly"""
        parser = cssutils.CSSParser()
        sheet = parser.parseString(css_content)
        rules = list(sheet.cssRules)

        # Randomly split rules into 2-4 files
        num_splits = random.randint(2, 4)
        random.shuffle(rules)
        splits = [[] for _ in range(num_splits)]

        for i, rule in enumerate(rules):
            splits[i % num_splits].append(rule)

        # Create split files
        main_file = original_path.parent / \
            f"{original_path.stem}_{self.build_id}.css"
        import_content = ""

        for i, split_rules in enumerate(splits):
            if i == 0:
                # First split goes in main file
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(str(rule) for rule in split_rules))
            else:
                # Create additional split files
                split_name = f"{original_path.stem}_split{i}_{self.build_id}.css"
                split_path = original_path.parent / split_name

                with open(split_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(str(rule) for rule in split_rules))

                import_content += f"@import url('./{split_name}');\n"

        # Add imports to main file
        if import_content:
            with open(main_file, 'a', encoding='utf-8') as f:
                f.write('\n' + import_content)

        return main_file

    def _process_js(self, file_path: Path) -> Path:
        """
        Transform JavaScript files with splitting and fingerprinting
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if self.config['resource_fingerprinting']['split_resources']:
                # Simple module splitting based on natural breaks
                chunks = self._split_js_content(content)

                # Create main file with imports
                new_name = f"{file_path.stem}_{self.build_id}.js"
                main_path = file_path.parent / new_name

                import_statements = []
                for i, chunk in enumerate(chunks[1:], 1):
                    chunk_name = f"{file_path.stem}_chunk{i}_{self.build_id}.js"
                    chunk_path = file_path.parent / chunk_name
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)
                    import_statements.append(f'import "./{chunk_name}";')

                # Write main file with imports and first chunk
                with open(main_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(import_statements) + '\n\n' + chunks[0])

                return main_path
            else:
                # Simple fingerprinting without splitting
                new_name = f"{file_path.stem}_{self.build_id}.js"
                output_path = file_path.parent / new_name
                shutil.copy2(file_path, output_path)
                return output_path

        except Exception as e:
            self.logger.error(
                f"Error processing JavaScript file {file_path}: {str(e)}")
            return file_path

    def _split_js_content(self, content: str) -> List[str]:
        """Helper method to split JavaScript content into chunks"""
        # Split on natural boundaries like function declarations and class definitions
        split_patterns = [
            r'(?=\n\s*function\s+\w+\s*\()',  # Function declarations
            r'(?=\n\s*class\s+\w+\s*\{)',     # Class declarations
            r'(?=\n\s*const\s+\w+\s*=\s*function)',  # Function expressions
            r'(?=\n\s*export\s+)'             # Export statements
        ]

        pattern = '|'.join(split_patterns)
        chunks = re.split(pattern, content)

        # Ensure we don't create too many small chunks
        min_chunk_size = 1000  # characters
        merged_chunks = []
        current_chunk = ''

        for chunk in chunks:
            if len(current_chunk) + len(chunk) < min_chunk_size:
                current_chunk += chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks if merged_chunks else [content]

    def _process_svg(self, file_path: Path) -> Path:
        """
        Transform SVG files with ID randomization and optimization
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Randomize IDs and classes
            content = self._randomize_svg_identifiers(content)

            # Create fingerprinted filename
            new_name = f"{file_path.stem}_{self.build_id}.svg"
            output_path = file_path.parent / new_name

            # Write transformed content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return output_path
        except Exception as e:
            self.logger.error(
                f"Error processing SVG file {file_path}: {str(e)}")
            return file_path

    def _randomize_svg_identifiers(self, content: str) -> str:
        """Helper method to randomize SVG IDs and classes"""
        # Find all id="..." and class="..." attributes
        id_pattern = r'id="([^"]*)"'
        class_pattern = r'class="([^"]*)"'

        id_map = {}
        class_map = {}

        def replace_id(match):
            original_id = match.group(1)
            if original_id not in id_map:
                id_map[original_id] = f"id_{self.build_id}_{len(id_map)}"
            return f'id="{id_map[original_id]}"'

        def replace_class(match):
            original_class = match.group(1)
            if original_class not in class_map:
                class_map[original_class] = f"c_{self.build_id}_{len(class_map)}"
            return f'class="{class_map[original_class]}"'

        content = re.sub(id_pattern, replace_id, content)
        content = re.sub(class_pattern, replace_class, content)

        return content

    def get_resource_map(self) -> Dict[str, str]:
        """Return mapping of original to transformed resource paths"""
        return self.resource_map

    def get_content_hashes(self) -> Dict[str, str]:
        """Return content hashes for each processed resource"""
        return self.content_hashes

    def _add_timestamp_variations(self) -> Dict[str, str]:
        """
        Adds timestamp-based variations to resources:
        - Random build timestamps
        - Cache-busting parameters
        - Time-based resource versioning
        """
        timestamps = {}
        for path in self.resource_map.values():
            if isinstance(path, Path):
                timestamp = int(time.time() * 1000)
                timestamps[str(path)] = f"?t={timestamp}"
        return timestamps

    def _create_encoding_variations(self, content: bytes) -> Dict[str, bytes]:
        """
        Creates different encoding versions of resources:
        - gzip compression
        - brotli compression
        - custom encoding schemes
        """
        variations = {}
        # Gzip compression
        gzip_content = gzip.compress(content)
        variations['gzip'] = gzip_content

        # Brotli compression if available
        try:
            import brotli
            br_content = brotli.compress(content)
            variations['br'] = br_content
        except ImportError:
            pass

        return variations

    def _apply_resource_chunking(self, content: bytes, chunk_size: int = 1024 * 1024) -> List[bytes]:
        """
        Splits resources into random-sized chunks:
        - Variable chunk sizes
        - Content-dependent splitting
        - Maintains resource integrity
        """
        chunks = []
        size = len(content)
        offset = 0

        while offset < size:
            chunk_size = random.randint(chunk_size // 2, chunk_size)
            chunks.append(content[offset:offset + chunk_size])
            offset += chunk_size

        return chunks

    def _generate_resource_map_file(self) -> Path:
        """
        Creates mapping file for resource transformations:
        - Original to transformed paths
        - Resource dependencies
        - Integrity hashes
        """
        map_data = {
            'resources': self.resource_map,
            'hashes': self.content_hashes,
            'build_id': self.build_id,
            'timestamp': time.time()
        }

        output_path = self.base_dir / f"resource-map_{self.build_id}.json"
        with open(output_path, 'w') as f:
            json.dump(map_data, f, indent=2)

        return output_path

    def _create_resource_dependencies(self) -> Dict[str, List[str]]:
        """
        Maps dependencies between resources:
        - CSS imports
        - Script dependencies
        - Asset relationships
        """
        dependencies = {}
        for path, transformed in self.resource_map.items():
            if path.endswith('.css'):
                deps = self._analyze_css_dependencies(path)
                dependencies[str(path)] = deps
            elif path.endswith('.js'):
                deps = self._analyze_js_dependencies(path)
                dependencies[str(path)] = deps
        return dependencies

    def _verify_transformations(self, original: Path, transformed: Path) -> bool:
        """
        Validates transformation reversibility:
        - Content integrity
        - Format compatibility
        - Resource accessibility
        """
        if not transformed.exists():
            return False

        orig_hash = self._calculate_content_hash(original)
        trans_hash = self._calculate_content_hash(transformed)

        # For transformed resources, verification depends on type
        if transformed.suffix in ['.jpg', '.png', '.webp']:
            return self._verify_image_transformation(original, transformed)

        return orig_hash == trans_hash

    def _cleanup_temporary_resources(self):
        """
        Removes intermediate transformation files:
        - Temporary chunks
        - Intermediate formats
        - Build artifacts
        """
        temp_pattern = f"*_{self.build_id}_temp*"
        for temp_file in self.base_dir.glob(temp_pattern):
            temp_file.unlink()

    def _apply_random_compression(self, file_path: Path) -> Path:
        """
        Applies varying compression levels:
        - Random quality settings
        - Different compression algorithms
        - Format-specific optimizations
        """
        output_path = file_path.parent / \
            f"{file_path.stem}_compressed_{self.build_id}{file_path.suffix}"

        if file_path.suffix in ['.jpg', '.jpeg']:
            quality = random.randint(60, 95)
            img = Image.open(file_path)
            img.save(output_path, quality=quality, optimize=True)
        elif file_path.suffix == '.png':
            img = Image.open(file_path)
            img.save(output_path, optimize=True)

        return output_path

    def _generate_font_variations(self, font_path: Path) -> List[Path]:
        """
        Creates font format variations:
        - WOFF/WOFF2 conversion
        - Subset generation
        - Variable font instances

        Note: Full font processing requires FontTools library.
        Without it, only basic file copying is performed.
        """
        variations = []
        try:
            # Try to import FontTools
            try:
                import fontTools.ttLib as ttlib
                HAS_FONTTOOLS = True
            except ImportError:
                HAS_FONTTOOLS = False
                self.logger.warning(
                    "FontTools not available. Font processing will be limited.")

            if HAS_FONTTOOLS:
                # Full font processing with FontTools
                font = ttlib.TTFont(font_path)
                output_formats = [
                    'woff', 'woff2'] if font_path.suffix == '.ttf' else ['ttf']

                for format in output_formats:
                    output_path = font_path.parent / \
                        f"{font_path.stem}_{self.build_id}.{format}"
                    font.save(str(output_path))
                    variations.append(output_path)
            else:
                # Basic file copying without format conversion
                new_path = font_path.parent / \
                    f"{font_path.stem}_{self.build_id}{font_path.suffix}"
                shutil.copy2(font_path, new_path)
                variations.append(new_path)

            return variations
        except Exception as e:
            self.logger.error(
                f"Error processing font file {font_path}: {str(e)}")
            return [font_path]

    def _create_resource_fallbacks(self, resource_path: Path) -> List[Path]:
        """
        Generates fallback versions:
        - Lower quality alternatives
        - Format conversions
        - Simplified versions
        """
        fallbacks = []
        if resource_path.suffix in ['.jpg', '.png', '.webp']:
            # Create lower quality versions
            img = Image.open(resource_path)
            for quality in [75, 50, 25]:
                fallback_path = resource_path.parent / \
                    f"{resource_path.stem}_fallback_{quality}_{self.build_id}{resource_path.suffix}"
                img.save(fallback_path, quality=quality)
                fallbacks.append(fallback_path)
        return fallbacks

    def _obfuscate_file_metadata(self, file_path: Path) -> None:
        """
        Modifies file metadata:
        - Creation/modification times
        - EXIF data removal
        - Permission changes
        """
        random_time = time.time() - random.randint(0, 30*24*60 *
                                                   60)  # Random time in last 30 days
        os.utime(file_path, (random_time, random_time))

        if file_path.suffix in ['.jpg', '.jpeg']:
            img = Image.open(file_path)
            data = list(img.getdata())
            img_no_exif = Image.new(img.mode, img.size)
            img_no_exif.putdata(data)
            img_no_exif.save(file_path)

    def _generate_resource_integrity(self, file_path: Path) -> str:
        """
        Generates integrity verification:
        - SRI hash generation
        - Custom integrity tokens
        - Verification metadata
        """
        algorithms = ['sha256', 'sha384', 'sha512']
        chosen_algo = random.choice(algorithms)

        hasher = hashlib.new(chosen_algo)
        with open(file_path, 'rb') as f:
            hasher.update(f.read())

        integrity = base64.b64encode(hasher.digest()).decode()
        return f"{chosen_algo}-{integrity}"

    def _randomize_file_paths(self, file_path: Path) -> Path:
        """Randomizes file paths while maintaining references"""
        random_name = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=8))
        new_path = file_path.parent / \
            f"{random_name}_{self.build_id}{file_path.suffix}"
        shutil.move(file_path, new_path)
        self.resource_map[str(file_path)] = str(new_path)
        return new_path

    def _mix_resource_formats(self) -> Dict[str, List[str]]:
        """Switches between equivalent resource formats"""
        format_variations = {}
        for orig_path in self.resource_map.keys():
            if orig_path.endswith('.css'):
                # Convert between CSS and inline styles
                format_variations[orig_path] = self._create_css_variations(
                    Path(orig_path))
            elif orig_path.endswith(('.jpg', '.png', '.webp')):
                # Convert between image formats
                format_variations[orig_path] = self._create_image_variations(
                    Path(orig_path))
        return format_variations

    def _generate_cache_variations(self) -> Dict[str, str]:
        """Creates different caching strategies"""
        cache_strategies = {}
        for path in self.resource_map.values():
            if isinstance(path, str):
                strategies = [
                    f"{path}?v={self.build_id}",
                    f"{path}?t={int(time.time())}",
                    f"{path}?nocache={random.randint(1000, 9999)}"
                ]
                cache_strategies[path] = random.choice(strategies)
        return cache_strategies

    def _create_css_variations(self, file_path: Path) -> List[str]:
        """Creates variations of CSS styles (inline vs external)"""
        variations = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                css_content = f.read()

            # Create inline style variation
            variations.append(f"<style>{css_content}</style>")

            # Create external file variation
            new_path = file_path.parent / \
                f"{file_path.stem}_var_{self.build_id}.css"
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(css_content)
            variations.append(str(new_path))

            return variations
        except Exception as e:
            self.logger.error(f"Error creating CSS variations: {str(e)}")
            return [str(file_path)]

    def _create_image_variations(self, file_path: Path) -> List[str]:
        """Creates variations of images in different formats"""
        variations = []
        try:
            img = Image.open(file_path)
            formats = ['png', 'jpeg', 'webp'] if img.mode == 'RGBA' else [
                'jpeg', 'webp']

            for fmt in formats:
                new_path = file_path.parent / \
                    f"{file_path.stem}_var_{self.build_id}.{fmt}"
                img.save(new_path, format=fmt, quality=85)
                variations.append(str(new_path))

            return variations
        except Exception as e:
            self.logger.error(f"Error creating image variations: {str(e)}")
            return [str(file_path)]

    def _add_resource_metadata(self, file_path: Path) -> Dict[str, str]:
        """Adds random but valid metadata to resources"""
        metadata = {
            'generator': f'Build_{self.build_id}',
            'build-timestamp': str(int(time.time())),
            'content-version': f'{random.randint(1,100)}.{random.randint(0,99)}',
            'resource-id': hashlib.md5(str(file_path).encode()).hexdigest()[:12]
        }

        if file_path.suffix in ['.jpg', '.jpeg', '.png']:
            img = Image.open(file_path)
            exif = img.getexif()
            for key, value in metadata.items():
                exif[key] = value
            img.save(file_path, exif=exif)

        return metadata

    def _handle_conditional_loading(self) -> List[str]:
        """Creates conditional resource loading variations"""
        variations = []
        for resource in self.resource_map.values():
            if isinstance(resource, str):
                conditions = [
                    f'<link rel="preload" href="{resource}" as="style" media="(min-width: 768px)">',
                    f'<link rel="prefetch" href="{resource}" media="print">',
                    f'<script defer src="{resource}" data-condition="async"></script>'
                ]
                variations.append(random.choice(conditions))
        return variations

    def _generate_build_id(self) -> str:
        """Generate unique build identifier"""
        timestamp = str(time.time()).encode()
        random_salt = str(random.getrandbits(32)).encode()
        return hashlib.md5(timestamp + random_salt).hexdigest()[:8]

    def _analyze_css_dependencies(self, path: str) -> List[str]:
        """
        Analyze CSS file for dependencies like imports and url() references
        """
        dependencies = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                css_content = f.read()

            # Find @import statements
            import_pattern = r'@import\s+[\'"]([^\'"]+)[\'"]'
            imports = re.findall(import_pattern, css_content)
            dependencies.extend(imports)

            # Find url() references
            url_pattern = r'url\([\'"]?([^\'"]+)[\'"]?\)'
            urls = re.findall(url_pattern, css_content)
            dependencies.extend(urls)

            # Remove duplicates and return
            return list(set(dependencies))
        except Exception as e:
            self.logger.error(
                f"Error analyzing CSS dependencies for {path}: {str(e)}")
            return []

    def _analyze_js_dependencies(self, path: str) -> List[str]:
        """
        Analyze JavaScript file for dependencies like imports and requires
        """
        dependencies = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                js_content = f.read()

            # Find import statements
            import_pattern = r'import.*?[\'"]([^\'"]+)[\'"]'
            imports = re.findall(import_pattern, js_content)
            dependencies.extend(imports)

            # Find require statements
            require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
            requires = re.findall(require_pattern, js_content)
            dependencies.extend(requires)

            return list(set(dependencies))
        except Exception as e:
            self.logger.error(
                f"Error analyzing JS dependencies for {path}: {str(e)}")
            return []

    def _verify_image_transformation(self, original: Path, transformed: Path) -> bool:
        """
        Verify image transformation validity by checking dimensions and basic properties
        """
        try:
            original_img = Image.open(original)
            transformed_img = Image.open(transformed)

            # Check if dimensions match (allowing for format conversion)
            dimensions_match = original_img.size == transformed_img.size

            # Check if image is valid and can be read
            transformed_img.verify()

            # Check if color mode is compatible
            color_mode_compatible = (
                original_img.mode == transformed_img.mode or
                (original_img.mode in ['RGB', 'RGBA']
                 and transformed_img.mode in ['RGB', 'RGBA'])
            )

            return dimensions_match and color_mode_compatible
        except Exception as e:
            self.logger.error(
                f"Error verifying image transformation: {str(e)}")
            return False

    def _update_resource_map_file(self):
        """Update the resource map file with final processing state"""
        try:
            map_data = {
                'resources': self.resource_map,
                'hashes': self.content_hashes,
                'metadata': self.metadata,
                'build_id': self.build_id,
                'timestamp': time.time(),
                'statistics': {
                    'total_processed': len(self.resource_map),
                    'by_type': {ext: len([p for p in self.resource_map.keys() if p.endswith(ext)])
                                for ext in ['.jpg', '.png', '.css', '.js', '.svg']}
                }
            }

            output_path = self.base_dir / f"resource-map_{self.build_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error updating resource map file: {str(e)}")

    def __del__(self):
        """Cleanup resources when object is destroyed"""
        try:
            self._cleanup_temporary_resources()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def _validate_path(self, file_path: Path) -> bool:
        """Validate file path exists and is accessible"""
        try:
            return (
                file_path.exists()
                and file_path.is_file()
                and os.access(file_path, os.R_OK)
            )
        except Exception as e:
            self.logger.error(f"Error validating path {file_path}: {str(e)}")
            return False

    def _validate_config(self, config: dict) -> bool:
        """Validate configuration dictionary"""
        required_keys = {
            "resource_fingerprinting": {
                "alternate_resource_loading",
                "randomize_image_format",
                "split_resources"
            }
        }

        try:
            if not all(key in config for key in required_keys):
                return False

            fingerprinting = config["resource_fingerprinting"]
            return all(key in fingerprinting for key in required_keys["resource_fingerprinting"])
        except Exception as e:
            self.logger.error(f"Error validating config: {str(e)}")
            return False

    def _check_requirements(self) -> bool:
        """Check if all required libraries are available"""
        required_libs = {
            'PIL': 'Pillow>=9.0.0',
            'cssutils': 'cssutils>=2.7.0'
        }

        missing_libs = []
        for lib, version in required_libs.items():
            try:
                __import__(lib)
            except ImportError:
                missing_libs.append(version)

        if missing_libs:
            self.logger.error(
                f"Missing required libraries: {', '.join(missing_libs)}")
            return False
        return True
