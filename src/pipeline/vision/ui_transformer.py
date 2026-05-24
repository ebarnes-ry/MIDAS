from typing import List, Tuple
from bs4 import BeautifulSoup, NavigableString
from .types import UIBlock, UIDocument

from .types import UIBlock, UIDocument
import re

class UITransformer:

    @staticmethod
    def transform_marker_json(marker_json_output) -> UIDocument:
        if marker_json_output is None:
            raise ValueError("Marker output is None - cannot transform")

        # The _collect_blocks function now returns both the UIBlock and its raw text
        collected_data: List[Tuple[UIBlock, str]] = []
        if hasattr(marker_json_output, 'children'):
            UITransformer._collect_blocks(marker_json_output.children, collected_data)

        # Sort based on the UIBlock's position
        sorted_data = sorted(collected_data, key=lambda item: (item[0].bbox[1], item[0].bbox[0]))
        
        # Unzip the sorted data
        sorted_blocks = [item[0] for item in sorted_data]
        raw_texts = [item[1] for item in sorted_data]

        # Build the full_page_text from the raw, unmodified text of each block.
        full_page_text = "\n\n".join(text for text in raw_texts if text)

        return UIDocument(
            blocks=sorted_blocks,
            full_page_text=full_page_text,
            images=getattr(marker_json_output, 'images', {}),
            metadata=getattr(marker_json_output, 'metadata', {}),
            dimensions=UITransformer._extract_dimensions(marker_json_output.children)
        )
    
    @staticmethod
    def _clean_html(html: str) -> str:
        if not html:
            return ""

        soup = BeautifulSoup(html, 'html.parser')

        for br in soup.find_all("br"):
            br.replace_with(NavigableString("\n"))

        for math_tag in soup.find_all("math"):
            tex = math_tag.get_text().strip()
            if not tex:
                math_tag.decompose()
                continue
            if math_tag.get("display") == "block":
                replacement = f"\n$${tex}$$\n"
            else:
                replacement = f"${tex}$"
            math_tag.replace_with(NavigableString(replacement))

        for block_tag in soup.find_all(["p", "div", "li"]):
            if block_tag.contents and not str(block_tag.contents[-1]).endswith("\n"):
                block_tag.append(NavigableString("\n"))

        text = soup.get_text()
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _collect_blocks(json_blocks: List, output_list: List[Tuple[UIBlock, str]]):
        if not json_blocks: return

        for json_block in json_blocks:
            if json_block is None: continue
            
            # html_content = getattr(json_block, 'html', '')
            # soup = BeautifulSoup(html_content, 'html.parser')
            # raw_text_content = (soup.get_text().strip() or None)
            html_content = getattr(json_block, 'html', '')

            soup = BeautifulSoup(html_content, 'html.parser')
            block_type_str = str(json_block.block_type).lower()

            # Get the complete text content of the block before we modify the soup.
            # This is the ground truth for what's in the block.
            # raw_text_content = soup.get_text().strip() or None
            raw_text_content = UITransformer._clean_html(html_content)
            
            # 1. Extract Image Description
            image_description = None
            if block_type_str in {'figure', 'picture', 'table', 'diagram'}:
                desc_tag = soup.find('p', attrs={'role': 'img'})
                if desc_tag:
                    desc_text = desc_tag.get_text(strip=True)
                    if 'description:' in desc_text.lower():
                        image_description = desc_text.split(':', 1)[1].strip()
                    # Decompose the tag so it's not included in the clean latex_content
                    desc_tag.decompose()

            # 2. Extract clean LaTeX Content (without the description)
            latex_content = UITransformer._clean_html(str(soup)) or None

            # 3. Determine if Editable
            is_editable = False
            always_editable_types = {'text', 'sectionheader', 'equation', 'inlinemath', 'code', 'caption', 'listitem', 'reference'}
            if block_type_str in always_editable_types:
                is_editable = True
            elif block_type_str in {'figure', 'picture', 'table', 'diagram'}:
                if not image_description and latex_content:
                    is_editable = True

            # 4. Create the UIBlock object
            flat_polygon = [] # Flatten polygon as before
            polygon = getattr(json_block, 'polygon', [])
            if polygon and isinstance(polygon[0], (list, tuple)):
                flat_polygon = [coord for point in polygon for coord in point]
            else:
                flat_polygon = polygon

            ui_block = UIBlock(
                id=str(json_block.id),
                block_type=str(json_block.block_type),
                html=html_content,
                polygon=flat_polygon,
                bbox=getattr(json_block, 'bbox', []),
                children=[],
                section_hierarchy=getattr(json_block, 'section_hierarchy', {}),
                images=getattr(json_block, 'images', {}),
                image_description=image_description,
                latex_content=latex_content,
                is_editable=is_editable
            )
            
            # Append the tuple of the clean block and its original raw text
            output_list.append((ui_block, raw_text_content))

            if hasattr(json_block, 'children') and json_block.children:
                UITransformer._collect_blocks(json_block.children, output_list)
            
    
    @staticmethod
    def _extract_dimensions(pages) -> Tuple[int, int]:
        if pages and len(pages) > 0 and hasattr(pages[0], 'bbox') and len(pages[0].bbox) >= 4:
            return (int(pages[0].bbox[2]), int(pages[0].bbox[3]))
        return (1000, 1200)

    # @staticmethod
    # def _collect_blocks(json_blocks: List, output_list: List[Tuple[UIBlock, str]]):
    #     for json_block in json_blocks:
    #         if json_block is None: continue
            
    #         html_content = getattr(json_block, 'html', '')
    #         soup = BeautifulSoup(html_content, 'html.parser')
    #         block_type_str = str(json_block.block_type).lower()

    #         image_description_tag = soup.find('p', attrs={'role': 'img'})
    #         if image_description_tag:
    #             print(f"[DEBUG] FOUND IMAGE DESCRIPTION TAG: {image_description_tag}")
    #             image_description_tag.decompose()

    #         for math_tag in soup.find_all('math'):
    #             is_display = math_tag.get('display') == 'block'
    #             tex_content = math_tag.get_text()
    #             replacement = f"$${tex_content}$$" if is_display else f"\\({tex_content}\\)"
    #             math_tag.replace_with(NavigableString(replacement))

    #         raw_text_content = soup.get_text().strip() or None

    #         desc_soup = BeautifulSoup(html_content, 'html.parser')
    #         desc_tag = desc_soup.find('p', attrs={'role': 'img'})
    #         image_description = None
    #         if desc_tag:
    #             desc_text = desc_tag.get_text(strip=True)
    #             if 'description:' in desc_text.lower():
    #                 image_description = desc_text.split(':', 1)[1].strip()

    #         #is_figure = block_type_str in {'figure', 'picture', 'table', 'diagram'}
            
    #         # Flatten polygon
    #         polygon = getattr(json_block, 'polygon', [])
    #         flat_polygon = [c for p in polygon for c in p] if polygon and isinstance(polygon[0], (list, tuple)) else polygon

    #         is_editable = (not is_figure and raw_text_content is not None) or (is_figure and image_description is not None)

    #         ui_block = UIBlock(
    #             id=str(json_block.id),
    #             block_type=str(json_block.block_type),
    #             html=html_content,
    #             polygon=flat_polygon,
    #             bbox=getattr(json_block, 'bbox', []),
    #             children=[],
    #             section_hierarchy=getattr(json_block, 'section_hierarchy', {}),
    #             images=getattr(json_block, 'images', {}),
    #             image_description=image_description,
    #             latex_content=raw_text_content,
    #             is_editable=is_editable,
    #         )
            
    #         output_list.append((ui_block, raw_text_content))

    #         if hasattr(json_block, 'children') and json_block.children:
    #             UITransformer._collect_blocks(json_block.children, output_list)
